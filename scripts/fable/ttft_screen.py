#!/usr/bin/env python3
"""Time-to-first-token A/B screen for coding-agent multi-turn traffic.

Why this exists
---------------
``scripts/fable/abba_window.py`` measures DECODE speed and
``scripts/fable/humaneval_screen.py`` measures quality. Neither measures the
number a coding agent actually feels on a warm turn: how long the server takes
to emit the first token when the conversation is long and the client re-rendered
part of the transcript.

The oMLX PR #3330 audit (``I-keepwarm-port-plan.md``) names that failure mode
exactly: a request whose long head is byte-identical to a banked turn but whose
tail was re-serialized by the client, so the *matching terminal* misses while
the *input prompt* is still an exact prefix. MTPLX already has the machinery for
it (``SessionBank.near_prefix_candidates`` ->
``restore_entry_prefix_cache`` -> boundary-true restore -> suffix-only prefill).
This screen measures whether the flags added alongside it move the number.

Three scenarios, in order, inside ONE conversation
--------------------------------------------------
1. ``cold``               -- ``POST /admin/cache/clear`` plus a per-repeat salt,
                             then the ~16K-token opening turn. Upper bound.
2. ``matching_terminal``  -- the same conversation with the model's own reply
                             and one more user turn appended. The banked
                             terminal is an exact prefix of this prompt: the
                             ordinary warm case, arm E in the audit.
3. ``rerendered_terminal``-- identical to (2) except the prior ASSISTANT turn is
                             re-rendered (bullet markers swapped, fence info
                             strings dropped, trailing whitespace added, tabs
                             expanded -- what an agent harness does when it
                             re-serializes a tool transcript). The divergence
                             lands INSIDE the banked terminal's token span, so
                             the terminal misses while the opening turn's prompt
                             is still an exact prefix. Arm D: the target.

Scenario 3 asserts that the re-render actually changed the text. A no-op
transform would silently turn arm D back into arm E.

Two measurement traps the audit names, and how this harness answers them
-----------------------------------------------------------------------
1. **The live-reference lease.** ``_session_keep_live_refs_for_request``
   (mtplx/server/openai.py) returns False for anonymous sessions with no tools,
   so a naive curl harness measures the snapshot-only path and UNDERSTATES the
   current baseline. This harness sends a session header on every request
   (``--live-ref header``, the default), which makes ``session_source`` start
   with ``header.`` and arms the lease. ``tools`` / ``both`` / ``env`` are the
   other three arming modes; the choice is recorded in the receipt and pinned
   into the arm identity so two arms can never differ in it.
2. **Session identity.** Header identity beats prompt-prefix inference
   (``mtplx/engine_session.py`` ``resolve_session_id``). The header names are
   listed in ``HEADER_SESSION_KEYS`` below and documented in
   ``scripts/fable/README.md`` -- they are NOT in ``docs/server.md``.

Arms
----
No ``--env`` at all is the control: production passes no ``MTPLX_*`` at all
(``start-qwen38-flash-next-210.sh`` exports only ``HF_HUB_OFFLINE``), so the
control arm must too -- see ``PRODUCTION_FAMILY_ENV``.

    --env MTPLX_FABLE_QSA_RESTORE_STAGING=1   restored-QSA capacity staging
    --env MTPLX_FABLE_PROTECTED_TERMINAL=1    protected-terminal eviction order
    --gdn-boundary-max 32                     pure env knob, raises
                                              MTPLX_GDN_BOUNDARY_MAX from 8

The audit's own recommendation is to run the boundary knob FIRST: it is a
one-env-var experiment costing MB against ~24 GiB of headroom and it attacks
the actual failure (``NO_SNAPSHOT_COVERAGE``) directly.

Run it THROUGH ``bench/laguna/run_guarded.py``. ``--dry-run`` prints the exact
outer command line; ``scripts/fable/README.md`` has the arm table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FABLE = ROOT / "scripts" / "fable"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

OUT_DIR = ROOT / ".benchmark-artifacts" / "fable" / "ttft"
RUN_GUARDED = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py"
)
QWEN_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")

DEFAULT_PORT = 8092
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_TOKENS = 192
DEFAULT_REPEATS = 3
DEFAULT_PROMPT_TOKENS = 16_384
#: Conservative bytes-per-token for the synthetic Python corpus below. Only
#: used to SIZE the prompt; the receipt always reports the server's own
#: ``usage.prompt_tokens``, and --min-prompt-tokens is the hard gate.
CHARS_PER_TOKEN = 3.4

SCENARIOS = ("cold", "matching_terminal", "rerendered_terminal")

#: Session-identity headers, in the order ``EngineSessionManager.resolve_session_id``
#: checks them (mtplx/engine_session.py). Header identity beats prompt-prefix
#: inference and arms the live-reference lease (``source.startswith("header.")``
#: in ``_session_keep_live_refs_for_request``). Undocumented in docs/server.md;
#: documented in scripts/fable/README.md.
HEADER_SESSION_KEYS = (
    "x-mtplx-session-id",
    "x-session-affinity",
    "x-session-id",
    "x-openwebui-chat-id",
    "x-openwebui-user-id",
)

#: The subset this harness sends. OpenCode stamps x-session-affinity and
#: x-session-id on every request; x-mtplx-session-id is MTPLX's own name and is
#: checked first.
SENT_SESSION_HEADERS = ("x-mtplx-session-id", "x-session-affinity", "x-session-id")

#: Coding-agent tool names ``_anonymous_coding_agent_tool_request`` matches.
#: Only sent in --live-ref tools/both.
AGENT_TOOL_NAMES = ("bash", "edit", "glob", "grep", "read", "todowrite", "write")

#: Production passes NO MTPLX_* environment at all. Stating that as an empty
#: mapping is not laziness: humaneval_screen's CONTROL_FAMILY_ENV describes the
#: ABBA *candidate* lane, and exporting it here would measure a lane :8080 does
#: not run. The server's own qwen4_exp family defaults are the control.
PRODUCTION_FAMILY_ENV: dict[str, str] = {}

LIVE_REF_ENV_KEY = "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS"
GDN_BOUNDARY_MAX_KEY = "MTPLX_GDN_BOUNDARY_MAX"


# --------------------------------------------------------------------------
# Shared implementation (one copy, borrowed from humaneval_screen)
# --------------------------------------------------------------------------


def _load_sibling(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_screen() -> Any:
    """humaneval_screen is import-safe (evalplus and MLX are both deferred)."""

    return _load_sibling("_fable_humaneval_screen", FABLE / "humaneval_screen.py")


# --------------------------------------------------------------------------
# Pure prompt construction (unit-tested, no GPU, no network)
# --------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a coding agent working inside a repository. Answer briefly and "
    "concretely, in markdown."
)

TURN_ONE_INSTRUCTION = (
    "Read the workspace dump above. Reply with a SHORT markdown report: one "
    "sentence of summary, then a bulleted list of exactly three observations, "
    "then one fenced python code block with a two-line fix. Nothing else."
)

TURN_TWO_INSTRUCTION = (
    "Now do the same for the second module in the dump, in the same format."
)


def synthetic_workspace(*, target_tokens: int, salt: str) -> str:
    """A deterministic ~``target_tokens``-token workspace dump.

    Seeded from ``salt`` so every repeat is a genuinely COLD prompt (the audit's
    salted-prompt contract, docs/benchmarking.md) while staying reproducible
    from the receipt: same salt in, same bytes out.
    """

    rng = random.Random(f"mtplx-ttft-screen:{salt}")
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    verbs = ["load", "merge", "resolve", "flush", "encode", "trim", "restore"]
    nouns = ["cache", "entry", "boundary", "token", "session", "prefix", "chunk"]
    lines: list[str] = [f"# workspace dump {salt}", ""]
    size = len(lines[0]) + 1
    index = 0
    while size < target_chars:
        verb = rng.choice(verbs)
        noun = rng.choice(nouns)
        block = [
            f"def {verb}_{noun}_{index}(state, *, limit={rng.randint(2, 4096)}):",
            f'    """{verb.capitalize()} the {noun} for shard {index}."""',
            f"    items = [item for item in state.{noun}s if item.id % 7 == {index % 7}]",
            f"    if len(items) > limit:",
            f"        items = items[:limit]",
            f"    return {{'{noun}': items, 'shard': {index}, 'verb': '{verb}'}}",
            "",
        ]
        for line in block:
            lines.append(line)
            size += len(line) + 1
        index += 1
    return "\n".join(lines)


def rerender_transcript(text: str) -> str:
    """Re-serialize an assistant/tool transcript the way agent harnesses do.

    Whitespace and markdown differences only -- the information is identical,
    the bytes are not, so the tokenized prompt diverges INSIDE the banked
    terminal while the opening turn's prompt stays an exact prefix.
    """

    out: list[str] = []
    for line in str(text).split("\n"):
        line = line.replace("\t", "    ").rstrip()
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("- "):
            stripped = "* " + stripped[2:]
        elif stripped.startswith("```") and len(stripped) > 3:
            stripped = "```"
        line = indent + stripped
        if line:
            line = line + " "
        out.append(line)
    return "\n".join(out)


def build_conversation(
    *,
    scenario: str,
    workspace: str,
    assistant_turn: str | None,
) -> list[dict[str, str]]:
    """The message list for one scenario. Pure; no I/O."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": f"{workspace}\n\n{TURN_ONE_INSTRUCTION}",
        },
    ]
    if scenario == "cold":
        return messages
    if not assistant_turn:
        raise ValueError(f"{scenario} needs the assistant turn from the cold scenario")
    prior = (
        assistant_turn
        if scenario == "matching_terminal"
        else rerender_transcript(assistant_turn)
    )
    if scenario == "rerendered_terminal" and prior == assistant_turn:
        raise ValueError(
            "the re-render transform was a no-op: scenario "
            "rerendered_terminal would silently measure matching_terminal"
        )
    messages.append({"role": "assistant", "content": prior})
    messages.append({"role": "user", "content": TURN_TWO_INSTRUCTION})
    return messages


def build_chat_payload(
    messages: Sequence[Mapping[str, str]],
    *,
    model_id: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": True,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": f"{name} tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in tools
        ]
    return payload


def session_headers(session_id: str, *, live_ref: str) -> dict[str, str]:
    """Identity headers for one conversation. See HEADER_SESSION_KEYS."""

    if live_ref not in {"header", "tools", "both", "env"}:
        raise ValueError(f"unknown --live-ref mode: {live_ref!r}")
    if live_ref in {"header", "both"}:
        return {key: session_id for key in SENT_SESSION_HEADERS}
    return {}


def request_tools(live_ref: str) -> tuple[str, ...]:
    return AGENT_TOOL_NAMES if live_ref in {"tools", "both"} else ()


def build_candidate_env(
    settings: Sequence[str],
    *,
    gdn_boundary_max: int | None,
    live_ref: str,
    parse: Any,
) -> dict[str, str]:
    """``--env`` plus the two knobs this harness owns. Fail-closed on conflict."""

    candidate = dict(parse(settings))
    if gdn_boundary_max is not None:
        if GDN_BOUNDARY_MAX_KEY in candidate:
            raise ValueError(
                f"--gdn-boundary-max conflicts with --env {GDN_BOUNDARY_MAX_KEY}=..."
            )
        if int(gdn_boundary_max) < 1:
            raise ValueError("--gdn-boundary-max must be >= 1")
        candidate[GDN_BOUNDARY_MAX_KEY] = str(int(gdn_boundary_max))
    if live_ref == "env":
        if LIVE_REF_ENV_KEY in candidate:
            raise ValueError(f"--live-ref env conflicts with --env {LIVE_REF_ENV_KEY}=...")
        candidate[LIVE_REF_ENV_KEY] = "1"
    elif LIVE_REF_ENV_KEY in candidate:
        raise ValueError(
            f"--env {LIVE_REF_ENV_KEY} changes which lane is measured; use "
            "--live-ref env so the receipt records it"
        )
    return candidate


def build_server_argv(
    *,
    python: str | Path,
    model: str | Path,
    model_id: str,
    host: str,
    port: int,
    ssd_session_cache: str,
) -> list[str]:
    """Production's server shape, with the three things a screen must pin.

    ``start-qwen38-flash-next-210.sh`` runs ``mtplx serve --model --model-id
    --host --port`` and defaults everything else. The profile/mode/depth/
    scheduler flags below are those same defaults stated explicitly so the
    receipt records the lane instead of implying it (and ``/health`` is
    asserted against them). What is NOT production: reasoning off, a greedy
    sampler. Production serves reasoning_effort=xhigh; a TTFT screen with
    thinking on would measure the think channel, not first-token latency.

    ``--ssd-session-cache`` stays ON by default because production's does
    (openai.py's ``--ssd-session-cache`` default, flipped on at v2.0.0) and the
    session cache is the subject of the measurement.
    """

    if ssd_session_cache not in {"on", "off"}:
        raise ValueError("--ssd-session-cache must be 'on' or 'off'")
    return [
        str(python),
        "-m",
        "mtplx.server.openai",
        "--model",
        str(model),
        "--model-id",
        str(model_id),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--profile",
        "turbo",
        "--generation-mode",
        "mtp",
        "--load-mtp",
        "--depth",
        "3",
        "--scheduler-mode",
        "serial",
        "--reasoning-mode",
        "off",
        "--temperature",
        "0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--ssd-session-cache",
        str(ssd_session_cache),
        "--no-auth",
    ]


def assert_server_contract(
    health: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    """Refuse to measure a server that is not the configuration we asked for.

    Parameterized on ``model_id`` rather than reusing humaneval_screen's copy,
    which closes over its own module constant and would pass silently for the
    wrong pack under ``--model-id``.
    """

    problems: list[str] = []

    def require(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, observed {actual!r}")

    require("health.model", health.get("model"), model_id)
    require("health.generation_mode", health.get("generation_mode"), "mtp")
    require("health.mtp_enabled", health.get("mtp_enabled"), True)
    require("health.depth", health.get("depth"), 3)
    require("health.profile", (health.get("profile") or {}).get("name"), "turbo")
    require("health.scheduler", (health.get("scheduler") or {}).get("mode"), "serial")
    require("settings.reasoning", settings.get("reasoning"), "off")
    require("settings.enable_thinking", settings.get("enable_thinking"), False)
    require("settings.generation_mode", settings.get("generation_mode"), "mtp")
    require("settings.depth", settings.get("depth"), 3)
    if problems:
        raise RuntimeError("server contract mismatch: " + "; ".join(problems))


def outer_command_line(
    *,
    label: str,
    candidate_env: Sequence[str] = (),
    gdn_boundary_max: int | None = None,
    repeats: int = DEFAULT_REPEATS,
    port: int = DEFAULT_PORT,
    child_timeout_seconds: int = 5400,
) -> str:
    """The exact guarded outer command for one arm (also printed by --dry-run)."""

    import shlex

    child = [
        str(VENV_PYTHON),
        str(FABLE / "ttft_screen.py"),
        "--label",
        label,
        "--repeats",
        str(int(repeats)),
        "--port",
        str(int(port)),
    ]
    for setting in candidate_env:
        child += ["--env", setting]
    if gdn_boundary_max is not None:
        child += ["--gdn-boundary-max", str(int(gdn_boundary_max))]
    outer = [
        "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python",
        str(RUN_GUARDED),
        "--plist",
        str(QWEN_PLIST),
        "--lock-timeout-seconds",
        "3600",
        "--timeout-seconds",
        "900",
        "--child-timeout-seconds",
        str(int(child_timeout_seconds)),
        "--",
        *child,
    ]
    return shlex.join(outer)


# --------------------------------------------------------------------------
# Streaming measurement
# --------------------------------------------------------------------------


def parse_sse_line(raw: bytes | str) -> dict[str, Any] | str | None:
    """One SSE line -> a chunk dict, the ``[DONE]`` sentinel, or None."""

    line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    line = line.strip()
    if not line.startswith("data:"):
        return None
    body = line[len("data:") :].strip()
    if not body:
        return None
    if body == "[DONE]":
        return "[DONE]"
    return json.loads(body)


def chunk_content(chunk: Mapping[str, Any]) -> tuple[str, str]:
    """``(content, reasoning_content)`` deltas from one chunk, never None."""

    choices = chunk.get("choices") or []
    if not choices:
        return "", ""
    delta = choices[0].get("delta") or {}
    return str(delta.get("content") or ""), str(delta.get("reasoning_content") or "")


def summarize_stream(
    events: Sequence[tuple[float, Any]],
) -> dict[str, Any]:
    """Fold ``(elapsed_s, parsed_chunk)`` events into one request receipt.

    Split out from the socket so the timing/SHA arithmetic is unit-testable
    without a server.
    """

    first_chunk_s: float | None = None
    first_token_s: float | None = None
    pieces: list[str] = []
    reasoning: list[str] = []
    final: dict[str, Any] | None = None
    total_s = 0.0
    for elapsed, chunk in events:
        total_s = max(total_s, float(elapsed))
        if first_chunk_s is None:
            first_chunk_s = float(elapsed)
        if chunk == "[DONE]" or not isinstance(chunk, dict):
            continue
        if chunk.get("mtplx_stats") is not None:
            final = chunk
        content, think = chunk_content(chunk)
        if think:
            reasoning.append(think)
        if content:
            if first_token_s is None:
                first_token_s = float(elapsed)
            pieces.append(content)
    text = "".join(pieces)
    stats = dict((final or {}).get("mtplx_stats") or {})
    usage = dict((final or {}).get("usage") or {})
    return {
        "client_first_chunk_s": first_chunk_s,
        "client_first_token_s": first_token_s,
        "client_total_s": total_s,
        "output_chars": len(text),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "reasoning_chars": len("".join(reasoning)),
        "text": text,
        "usage": usage,
        "finish_reason": (
            ((final or {}).get("choices") or [{}])[0].get("finish_reason")
        ),
        "server": {
            key: stats.get(key)
            for key in (
                "ttft_s",
                "prefill_tok_s",
                "decode_tok_s",
                "prompt_eval_time_s",
                "decode_elapsed_s",
                "server_elapsed_s",
                "cached_tokens",
                "new_prefill_tokens",
                "session_cache_hit",
                "cache_miss_reason",
                "session_restore_mode",
                "session_restore_served",
                "peak_memory_bytes",
                "accepted_by_depth",
                "drafted_by_depth",
            )
        },
    }


def stream_chat(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST a streaming completion and time the first SSE frame + first token."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **dict(headers),
        },
        method="POST",
    )
    events: list[tuple[float, Any]] = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            elapsed = time.perf_counter() - started
            parsed = parse_sse_line(raw)
            if parsed is None:
                continue
            events.append((elapsed, parsed))
            if parsed == "[DONE]":
                break
    return summarize_stream(events)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _quantile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_scenarios(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-scenario medians, p95 and parity, over every repeat.

    p95 is reported alongside the median on purpose: the b5fac4ac
    falsification was a TAIL result (27.6 s worst stall) invisible in medians.
    """

    summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario]
        if not subset:
            continue
        client = [
            row["client_first_token_s"]
            for row in subset
            if row.get("client_first_token_s") is not None
        ]
        server = [
            row["server"]["ttft_s"]
            for row in subset
            if (row.get("server") or {}).get("ttft_s") is not None
        ]
        shas = sorted({str(row["output_sha256"]) for row in subset})
        summary[scenario] = {
            "repeats": len(subset),
            "visible_ttft_s": {
                "median": statistics.median(client) if client else None,
                "min": min(client) if client else None,
                "max": max(client) if client else None,
                "p95": _quantile(client, 0.95),
            },
            "model_ttft_s": {
                "median": statistics.median(server) if server else None,
                "min": min(server) if server else None,
                "max": max(server) if server else None,
                "p95": _quantile(server, 0.95),
            },
            "prompt_tokens": sorted(
                {int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in subset}
            ),
            "cached_tokens": sorted(
                {int((row["server"] or {}).get("cached_tokens") or 0) for row in subset}
            ),
            "session_cache_hit": sorted(
                {bool((row["server"] or {}).get("session_cache_hit")) for row in subset}
            ),
            "session_restore_mode": sorted(
                {str((row["server"] or {}).get("session_restore_mode") or "") for row in subset}
            ),
            "cache_miss_reason": sorted(
                {str((row["server"] or {}).get("cache_miss_reason") or "") for row in subset}
            ),
            "output_sha256": shas,
            "output_deterministic": len(shas) == 1,
        }
    return summary


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def arm_identity(
    candidate_env: Mapping[str, str],
    server_argv: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Everything that must match for two receipts to be the SAME arm shape."""

    return {
        "candidate_env": dict(candidate_env),
        "production_family_env": dict(PRODUCTION_FAMILY_ENV),
        "server_argv": list(server_argv),
        "live_ref": str(args.live_ref),
        "repeats": int(args.repeats),
        "max_tokens": int(args.max_tokens),
        "prompt_tokens_target": int(args.prompt_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "model_id": str(args.model_id),
        "salt_seed": str(args.salt_seed),
    }


def run_repeat(
    *,
    base_url: str,
    args: argparse.Namespace,
    repeat: int,
    screen: Any,
) -> list[dict[str, Any]]:
    """One cold -> matching -> rerendered pass over a fresh conversation."""

    salt = f"{args.salt_seed}-r{repeat}"
    session_id = f"ttft-screen-{args.label}-{salt}"
    workspace = synthetic_workspace(target_tokens=args.prompt_tokens, salt=salt)
    headers = session_headers(session_id, live_ref=args.live_ref)
    tools = request_tools(args.live_ref)

    # Cold contract (docs/benchmarking.md): clear the bank AND salt the prompt.
    screen.http_post(f"{base_url}/admin/cache/clear", {}, timeout=120.0)

    rows: list[dict[str, Any]] = []
    assistant_turn: str | None = None
    assistant_sha = ""
    for scenario in SCENARIOS:
        messages = build_conversation(
            scenario=scenario,
            workspace=workspace,
            assistant_turn=assistant_turn,
        )
        payload = build_chat_payload(
            messages,
            model_id=args.model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tools=tools,
        )
        result = stream_chat(
            f"{base_url}/v1/chat/completions",
            payload,
            headers={
                **headers,
                "x-mtplx-request-id": f"ttft-{args.label}-{salt}-{scenario}",
            },
            timeout=args.request_timeout,
        )
        if result["reasoning_chars"]:
            raise RuntimeError(
                f"{scenario}: server emitted reasoning with thinking off -- the "
                "screen contract is broken, not the flag"
            )
        if not result["text"].strip():
            raise RuntimeError(f"{scenario}: empty completion; nothing to time")
        prompt_tokens = int((result["usage"] or {}).get("prompt_tokens") or 0)
        if scenario == "cold" and prompt_tokens < args.min_prompt_tokens:
            raise RuntimeError(
                f"cold prompt was {prompt_tokens} tokens, below "
                f"--min-prompt-tokens {args.min_prompt_tokens}: this would not "
                "be a long-context measurement"
            )
        if scenario == "cold":
            assistant_turn = result["text"]
            # Turns 2 and 3 embed this text, so two arms are only comparable
            # when it is identical. Recorded, not assumed.
            assistant_sha = hashlib.sha256(
                assistant_turn.encode("utf-8")
            ).hexdigest()
        row = {
            "repeat": int(repeat),
            "scenario": scenario,
            "salt": salt,
            "session_id": session_id,
            "messages": len(messages),
            "assistant_turn_sha256": assistant_sha,
            **{key: value for key, value in result.items() if key != "text"},
        }
        rows.append(row)
        print(
            f"[ttft-screen:{args.label}] r{repeat} {scenario} "
            f"visible={result['client_first_token_s']!r}s "
            f"model={(result['server'] or {}).get('ttft_s')!r}s "
            f"prompt={prompt_tokens} "
            f"cached={(result['server'] or {}).get('cached_tokens')} "
            f"hit={(result['server'] or {}).get('session_cache_hit')} "
            f"restore={(result['server'] or {}).get('session_restore_mode')!r}",
            flush=True,
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Candidate MTPLX_* env exported before the server starts "
            "(repeatable). No --env at all IS the control arm."
        ),
    )
    parser.add_argument(
        "--gdn-boundary-max",
        type=int,
        default=None,
        help=(
            f"Set {GDN_BOUNDARY_MAX_KEY} (default 8). A pure env knob: more "
            "recurrent boundaries per entry means a boundary-true restore can "
            "land closer to the divergence. Costs MB per boundary."
        ),
    )
    parser.add_argument(
        "--live-ref",
        choices=("header", "tools", "both", "env"),
        default="header",
        help=(
            "How the live-reference lease is armed. 'header' sends the session "
            "headers (default, and what OpenCode does); 'tools' sends an agent "
            "tool array; 'both' does both; 'env' sets "
            f"{LIVE_REF_ENV_KEY}=1. Never mix across arms -- it is pinned into "
            "the arm identity."
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--receipt-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--python", type=Path, default=VENV_PYTHON)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--min-prompt-tokens", type=int, default=12_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ssd-session-cache", choices=("on", "off"), default="on")
    parser.add_argument("--salt-seed", default=None)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--server-ready-timeout", type=float, default=1200.0)
    parser.add_argument("--warmup-timeout", type=float, default=900.0)
    parser.add_argument(
        "--guard-mode",
        choices=("auto", "attestation", "window"),
        default="auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the outer guarded command and the server argv, then exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    screen = load_screen()
    if args.model is None:
        args.model = screen.MODEL
    if args.model_id is None:
        args.model_id = screen.MODEL_ID
    if args.salt_seed is None:
        args.salt_seed = time.strftime("%Y%m%dT%H%M%S")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    candidate_env = build_candidate_env(
        args.env,
        gdn_boundary_max=args.gdn_boundary_max,
        live_ref=args.live_ref,
        parse=screen.parse_env_settings,
    )
    server_argv = build_server_argv(
        python=args.python,
        model=args.model,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        ssd_session_cache=args.ssd_session_cache,
    )

    if args.dry_run:
        print("[ttft-screen] outer command:")
        print("  " + outer_command_line(
            label=args.label,
            candidate_env=args.env,
            gdn_boundary_max=args.gdn_boundary_max,
            repeats=args.repeats,
            port=args.port,
        ))
        print("[ttft-screen] server argv:")
        print("  " + " ".join(server_argv))
        print("[ttft-screen] candidate env: " + json.dumps(candidate_env))
        print("[ttft-screen] session headers: " + ", ".join(SENT_SESSION_HEADERS))
        return 0

    driver = screen.load_abba_driver()
    guard = driver.acquire_guard(args.guard_mode)

    provenance = screen.model_provenance(args.model)
    if provenance["resolved_sha"] != screen.EXPECTED_MODEL_SHA:
        raise RuntimeError(
            f"model artifact moved: expected {screen.EXPECTED_MODEL_SHA}, "
            f"observed {provenance['resolved_sha']}"
        )
    if not screen.port_is_free(args.host, args.port):
        raise RuntimeError(
            f"{args.host}:{args.port} already has a listener; refusing to overlap"
        )

    available = driver.wait_for_memory()
    print(
        f"[ttft-screen] lock attested ({guard['mode']}); "
        f"reclaimable={available / 1024**3:.2f} GiB",
        flush=True,
    )

    run_dir = args.receipt_dir.resolve() / f"{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    receipt_path = run_dir.parent / f"{args.label}.json"
    arm_claim = screen.claim_arm_identity(
        run_dir, arm_identity(candidate_env, server_argv, args)
    )
    print(f"[ttft-screen] arm identity {arm_claim['claimed']}", flush=True)

    environment = screen.build_server_env(
        os.environ, candidate_env, family=PRODUCTION_FAMILY_ENV
    )
    # The server must import THIS worktree's mtplx, not whatever the venv has
    # installed. cwd puts ROOT first on sys.path for `-m`; PYTHONPATH makes it
    # explicit in the receipt.
    environment["PYTHONPATH"] = str(ROOT)
    base_url = f"http://{args.host}:{args.port}"
    timings: dict[str, float] = {}
    run_started = time.time()

    arm = "candidate" if candidate_env else "control"
    print(
        f"[ttft-screen] arm={arm} env={json.dumps(candidate_env, sort_keys=True)} "
        f"live_ref={args.live_ref}",
        flush=True,
    )
    print("[ttft-screen] " + " ".join(server_argv), flush=True)

    process: subprocess.Popen[Any] | None = None
    stop_receipt: dict[str, Any] = {}
    warmup: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    snapshot: dict[str, Any] = {}
    try:
        server_started = time.time()
        with log_path.open("wb") as log_handle:
            process = subprocess.Popen(
                server_argv,
                cwd=str(ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            health = screen.wait_for_health(
                base_url,
                process=process,
                log_path=log_path,
                timeout=args.server_ready_timeout,
            )
            settings = screen.http_get(f"{base_url}/v1/mtplx/settings", timeout=15.0)
            assert_server_contract(health, settings, model_id=args.model_id)

            background = (health.get("warmup") or {}).get("background")
            if isinstance(background, dict):
                warmup_health = screen.load_run_guarded().wait_for_background_warmup(
                    base_url,
                    timeout=args.warmup_timeout,
                    fetch=lambda url: screen.http_get(f"{url}/health", timeout=15.0),
                )
                warmup = {
                    "waited": True,
                    "state": (
                        ((warmup_health.get("warmup") or {}).get("background") or {})
                        .get("state")
                    ),
                }
            else:
                warmup = {
                    "waited": False,
                    "reason": "health has no warmup.background block",
                }
            timings["server_ready_s"] = time.time() - server_started
            print(
                f"[ttft-screen] ready in {timings['server_ready_s']:.0f}s",
                flush=True,
            )

            measurement_started = time.time()
            for repeat in range(1, int(args.repeats) + 1):
                rows.extend(
                    run_repeat(
                        base_url=base_url,
                        args=args,
                        repeat=repeat,
                        screen=screen,
                    )
                )
            timings["measurement_s"] = time.time() - measurement_started

            try:
                snapshot = screen.http_get(
                    f"{base_url}/v1/mtplx/snapshot", timeout=30.0
                )
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                snapshot = {}
    finally:
        if process is not None:
            stop_receipt = screen.stop_server(process)
            print(f"[ttft-screen] server stopped: {stop_receipt}", flush=True)

    timings["total_s"] = time.time() - run_started
    summary = summarize_scenarios(rows)
    receipt = {
        "schema": "mtplx-fable-ttft-screen-v1",
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm,
        "flags": {
            "candidate_env": dict(candidate_env),
            "production_family_env": dict(PRODUCTION_FAMILY_ENV),
            "never_exported": dict(screen.NEVER_EXPORT),
            "server_argv": list(server_argv),
            "pythonpath": str(ROOT),
        },
        "measurement": {
            "live_ref_mode": args.live_ref,
            "session_headers_sent": list(SENT_SESSION_HEADERS),
            "session_header_keys_recognised": list(HEADER_SESSION_KEYS),
            "tools_sent": list(request_tools(args.live_ref)),
            "scenarios": list(SCENARIOS),
            "repeats": int(args.repeats),
            "salt_seed": str(args.salt_seed),
            "prompt_tokens_target": int(args.prompt_tokens),
            "min_prompt_tokens": int(args.min_prompt_tokens),
            "max_tokens": int(args.max_tokens),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ssd_session_cache": args.ssd_session_cache,
        },
        "model": dict(provenance),
        "guard": dict(guard),
        "summary": summary,
        "requests": rows,
        "server_health": dict(health),
        "server_settings": dict(settings),
        "server_warmup": dict(warmup),
        "session_bank": (snapshot.get("session_bank") or {}),
        "server_log": {
            "path": str(log_path),
            "stop": stop_receipt,
            "tail": screen.tail_lines(log_path, 200),
        },
        "timings_s": dict(timings),
        "arm_identity": dict(arm_claim),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for scenario in SCENARIOS:
        block = summary.get(scenario)
        if not block:
            continue
        visible = block["visible_ttft_s"]
        model = block["model_ttft_s"]
        print(
            f"[ttft-screen] {args.label} {scenario}: "
            f"visible median={visible['median']!r}s p95={visible['p95']!r}s; "
            f"model median={model['median']!r}s p95={model['p95']!r}s; "
            f"cached={block['cached_tokens']} hit={block['session_cache_hit']} "
            f"restore={block['session_restore_mode']} "
            f"sha={block['output_sha256']}",
            flush=True,
        )
    print(f"wrote {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
