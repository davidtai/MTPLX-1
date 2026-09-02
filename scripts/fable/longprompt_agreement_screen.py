#!/usr/bin/env python3
"""Long-prompt greedy-agreement screen for PREFILL numerics changes.

Why this exists
---------------
``scripts/fable/humaneval_screen.py`` is the task-eval gate for DECODE-side
rounding-class kernels, and it cannot gate a prefill change: every HumanEval
prompt is a few hundred tokens, so it fits in ONE prefill chunk. A chunk-width
move (``MTPLX_PREFILL_CHUNK_SIZE=4096``), a blocked GDN prefill
(``MTPLX_GDN_BLOCKED_PREFILL=1``) or a QSA query tile
(``MTPLX_FABLE_PREFILL_QSA_QUERY_TILE``) changes the ORDER and SHAPE of the
reductions that build the KV cache. On a single-chunk prompt most of that code
is never reached, and the ones that are reached run on one chunk with no
boundary to get wrong. HumanEval passing therefore says nothing about them.

This screen puts real multi-chunk prompts through the same guarded one-arm
server that ``humaneval_screen`` uses and compares GREEDY continuations
between two arms token by token. Greedy decoding is a hash of the prefill: any
numeric drift big enough to flip one argmax shows up as a divergence, and
everything after it is permanently different. So the metric is not "token
agreement over 256 positions" -- it is the EXACT-MATCH PREFIX LENGTH, plus
(when the server can supply it) the evidence that the flip happened at a
near-tie rather than at a confident position.

Prompt lengths are chosen so the 2,048 and 4,096 chunk layouts cut them in
DIFFERENT places: 16,384 (8 x 2,048 vs 4 x 4,096), 9,216 (4 x 2,048 + 1,024 vs
2 x 4,096 + 1,024) and 4,608 (2 x 2,048 + 512 vs 1 x 4,096 + 512). Two of the
three carry a RAGGED final chunk, which the 16,384 cell -- a multiple of both
widths -- never exercises at all.

One property these sizes do NOT give: a final chunk of a different WIDTH in
the two layouts. Both tails above are 512-aligned, so the last chunk starts at
the same offset either way. Getting that as well needs ``n % 4096 >= 2048``
(7,168, say: 1,024 vs 3,072). Everything before the tail still differs -- the
boundary offsets and the chunk COUNT -- which is what the KV state a boundary
bug corrupts depends on.

Shape of one run
----------------
1. consume the canonical GPU guard evidence (``abba_driver.acquire_guard``),
2. wait for reclaimable memory, then start ONE MTPLX server on :8093 from this
   worktree's venv with the control family env plus ``--env`` candidates,
3. wait for ``/health`` + background warmup + a READY chat,
4. probe once whether this build supports chat ``logprobs``,
5. one greedy chat completion per prompt (temperature 0, thinking OFF,
   ``max_tokens`` 256, ``n`` 1),
6. stop the server and write a receipt under
   ``.benchmark-artifacts/fable/longprompt/``.

Then, with no GPU at all, ``--score CONTROL.json CANDIDATE.json`` compares two
receipts. A third path is read as a SECOND CONTROL and reported as the
run-to-run noise floor: greedy decoding is deterministic, so control-vs-control
that is not 100% identical voids the whole comparison.

logprobs
--------
``mtplx/server/openai.py`` REJECTS ``logprobs``/``top_logprobs`` on
``/v1/chat/completions`` with a 400 ("support is planned"), and the one place
that does emit per-position top-K -- ``/v1/completions`` with
``echo=true, logprobs=k, max_tokens=0`` -- is the wrong instrument here:
``mtplx/generation.py:score_prompt_logprobs`` teacher-forces the prompt in its
OWN fixed 256-token chunks, so it never reaches the prefill chunker or the
4,096-row QSA graph this screen is gating. Requesting it would produce
confident-looking numbers about code that did not run.

So this screen requests ``n=1`` greedy tokens only, and SAYS SO: the receipt
carries a ``logprobs`` block recording the probe result and the reason. The
scorer is written for both worlds -- it applies the full near-tie rule when a
receipt carries logprobs and the strict rule when it does not (see
the ``--score`` help text and the verdict rule in ``scripts/fable/README.md``).

Verdict rule
------------
With logprobs available:
  PASS  when every divergence sits at a NEAR-TIE (the control's top-2 margin
        at the divergence position is < ``--near-tie-nats`` (0.05) nats) AND
        the mean per-position top-1 logprob |delta| over the agreeing
        prefixes is < ``--max-logprob-delta`` (0.02).
  FLAG  otherwise, naming the prompt and position for human review.
Without logprobs (this build):
  PASS  only when every prompt agrees for its full length. There is no
        evidence that a divergence was a near-tie, so a divergence cannot be
        excused -- it is FLAGged for human review with the diverging text.

Run it THROUGH ``bench/laguna/run_guarded.py`` -- ``--print-commands`` prints
the three exact outer command lines (control, chunk-4096 candidate, blocked-GDN
candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FABLE = ROOT / "scripts" / "fable"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

OUT_DIR = ROOT / ".benchmark-artifacts" / "fable" / "longprompt"
RUN_GUARDED = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py"
)
QWEN_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")

DEFAULT_PORT = 8093  # 8091 = humaneval_screen, 8092 = ttft_screen
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TOP_LOGPROBS = 5

#: Sampler seeds in ``abba_window.PRODUCTION_SEEDS`` are 20260829..20260831.
#: The ABBA production cell is ONE fixed 16,384-token prompt
#: (``abba_driver.production_prompt_content``, pinned by SHA-256) and its seeds
#: vary the SAMPLER, not the prompt. Greedy decoding has no sampler seed, so
#: this screen reuses those integers as PROMPT-selection seeds instead: each
#: one rotates the pinned coding-context fixture by a different number of lines
#: before the same length-targeting builder cuts it. Same material, same
#: builder, six distinct prompts -- so one divergence cannot be a single
#: prompt's fluke.
LONG_PROMPT_SEEDS = (
    20260829,
    20260830,
    20260831,
    20260832,
    20260833,
    20260834,
)
LONG_PROMPT_TOKENS = 16_384

#: The two ragged-tail prompts: a multiple of NEITHER 2,048 nor 4,096, so each
#: one ends in a partial chunk and the two layouts put their boundaries in
#: different places. See the module docstring for the one property they do not
#: give (a final chunk of a different width).
SHORT_PROMPTS: tuple[tuple[str, int, int], ...] = (
    ("mid-9k", 20260835, 9_216),
    ("short-4k5", 20260836, 4_608),
)

#: How close the templated prompt must land to its target, in tokens. The
#: builder decodes a truncated id list back to text, so one context token can
#: be worth two at the seam and an exact hit is not always reachable: of the
#: eight shipped prompts, seven land EXACTLY on their target and 4,608
#: oscillates between 4,607 and 4,609. 8 accepts that and still refuses a
#: prompt that missed its chunk geometry. The receipt always records the
#: server's own ``usage.prompt_tokens`` as the authoritative length.
PROMPT_TOKEN_TOLERANCE = 8

INSTRUCTION_SUFFIX = (
    "\n\nAnswer with a SHORT markdown report: one sentence of summary, then a "
    "bulleted list of exactly three observations about the module above. "
    "No code."
)

#: Metal allocator caps, IDENTICAL to ``abba_driver.MEMORY_LIMIT_BYTES`` /
#: ``WIRED_LIMIT_BYTES``. Neither humaneval_screen nor ttft_screen sets these,
#: which is safe for their prompt sizes but NOT for this one: with the keys
#: unset the server falls back to ``_apply_metal_memory_caps``' default wired
#: cap (60% of 128 GiB = 76.8 GiB, raised to the qwen4_exp resident floor of
#: ~83.3 GiB), while a 16K prefill at chunk 4,096 peaks around 92.7 GB
#: (~86.3 GiB). That is over the default wired cap, and Metal's forced
#: eviction is the documented ~10x serve collapse -- the candidate arm would
#: be scored on a machine that was swapping.
#:
#: Setting the wired limit also ARMS ``mtplx.fable_prefill_chunk``'s
#: construction-time geometry guard, whose budget is exactly
#: ``MTPLX_WIRED_LIMIT_BYTES`` when no explicit budget is set
#: (``resolve_budget_bytes``): 90 GiB minus the 2 GiB guard margin = 88 GiB,
#: which the ~86.3 GiB projected peak clears. Unset, the guard is INERT and a
#: geometry that swaps would be admitted silently.
MEMORY_LIMIT_BYTES = 96 * 1024**3
WIRED_LIMIT_BYTES = 90 * 1024**3
MEMORY_ENV: dict[str, str] = {
    "MTPLX_MEMORY_LIMIT_BYTES": str(MEMORY_LIMIT_BYTES),
    "MTPLX_WIRED_LIMIT_BYTES": str(WIRED_LIMIT_BYTES),
}

#: Verdict thresholds. Stated here, in ``--help``, and in the README.
DEFAULT_NEAR_TIE_NATS = 0.05
DEFAULT_MAX_LOGPROB_DELTA = 0.02

#: The three arms this screen exists to compare, as ``--env`` settings.
#: ``MTPLX_PREFILL_CHUNK_SIZE`` and ``MTPLX_QSA_PREFILL_COMPILE_ROWS`` MUST
#: move together: ``mtplx/models/qwen4_exp.py`` only serves the compiled QSA
#: prefill graph when ``rows == _qsa_prefill_compile_rows()``, so moving the
#: width alone demotes every full chunk to the eager selector and the run
#: would score a lane nobody proposed. ``fable_prefill_chunk
#: .assert_prefill_chunk_coherent`` refuses that pair server-side; this screen
#: refuses it before the model loads (``assert_candidate_env_coherent``).
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("control", ()),
    (
        "chunk4096",
        (
            "MTPLX_PREFILL_CHUNK_SIZE=4096",
            "MTPLX_QSA_PREFILL_COMPILE_ROWS=4096",
        ),
    ),
    ("gdnblocked", ("MTPLX_GDN_BLOCKED_PREFILL=1",)),
)

PREFILL_CHUNK_KEY = "MTPLX_PREFILL_CHUNK_SIZE"
COMPILE_ROWS_KEY = "MTPLX_QSA_PREFILL_COMPILE_ROWS"
ALLOW_MISMATCH_KEY = "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH"


# --------------------------------------------------------------------------
# Shared implementation (one copy, borrowed from humaneval_screen -- the same
# pattern ttft_screen.py uses)
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
# Pure environment / argv construction (unit-tested, no GPU, no network)
# --------------------------------------------------------------------------


def assert_candidate_env_coherent(candidate: Mapping[str, str]) -> None:
    """Refuse a candidate the rest of the stack would silently mis-serve.

    Mirrors ``mtplx.fable_prefill_chunk.assert_prefill_chunk_coherent``, but
    fires here -- before a 92 GB model load and eight 16K prefills -- instead
    of inside the server.
    """

    chunk = str(candidate.get(PREFILL_CHUNK_KEY) or "").strip()
    rows = str(candidate.get(COMPILE_ROWS_KEY) or "").strip()
    if not chunk and not rows:
        return
    if str(candidate.get(ALLOW_MISMATCH_KEY) or "").strip() in ("1", "true", "yes", "on"):
        return
    if chunk and chunk.lower() in ("auto", "dense", "repage"):
        # Non-numeric widths resolve per layout; the compile-rows pairing is
        # only meaningful against a numeric width.
        return
    if chunk != rows:
        raise ValueError(
            f"{PREFILL_CHUNK_KEY}={chunk or '<unset>'} does not match "
            f"{COMPILE_ROWS_KEY}={rows or '<unset>'}: the QSA prefill graph "
            "bank only captures its own row width, so every full chunk would "
            "fall back to the eager selector and the arm would score a lane "
            f"nobody proposed. Set both, or {ALLOW_MISMATCH_KEY}=1."
        )


def build_screen_family_env(control_family: Mapping[str, str]) -> dict[str, str]:
    """humaneval_screen's stated ABBA control lane plus the Metal caps.

    The prefill candidates this screen gates are ABBA-lane candidates, and
    ``abba_window`` measures their SPEED against exactly this control family
    -- so the quality gate has to hold the same lane fixed or the two receipts
    describe different servers.
    """

    family = dict(control_family)
    overlap = sorted(set(family) & set(MEMORY_ENV))
    if overlap:
        raise RuntimeError(
            f"the control family already sets {overlap}; refusing to redefine "
            "the Metal caps behind it"
        )
    family.update(MEMORY_ENV)
    return family


def build_server_argv(
    *,
    python: str | Path,
    model: str | Path,
    model_id: str,
    host: str,
    port: int,
) -> list[str]:
    """The MTPLX OpenAI server command line for one screen arm.

    Byte-for-byte humaneval_screen's shape (production's launcher flags plus
    reasoning off / greedy / no SSD session cache), re-derived here rather
    than borrowed so a change to the quality screen's sampler cannot silently
    move this screen's lane.

    ``--ssd-session-cache off`` matters MORE here than it does there: a
    cross-request session cache could serve the second arm's 16K prefill out
    of the first arm's KV, which is the one thing that would make a broken
    prefill look identical.
    """

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
        "off",
        "--no-auth",
    ]


def assert_server_contract(
    health: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    """Refuse to screen a server that is not the configuration we asked for.

    Parameterized on ``model_id`` rather than reusing humaneval_screen's copy,
    which closes over its own module constant and would fail confusingly (or,
    for a renamed pack, pass for the wrong one) under ``--model-id``.
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


def build_chat_payload(
    prompt: str,
    *,
    model_id: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """One greedy, thinking-off chat completion.

    ``temperature`` is pinned to 0 and nothing else about the sampler can
    matter: ``_make_sampler`` in ``mtplx/server/openai.py`` returns a bare
    ``mx.argmax`` when ``temperature <= 0`` and no penalties are set, before
    top-p/top-k are ever consulted. That is what makes the continuation a
    deterministic function of the prefill.

    ``top_logprobs`` is only ever passed when the startup probe found this
    build accepts it; see the module docstring.
    """

    payload: dict[str, Any] = {
        "model": str(model_id),
        "messages": [{"role": "user", "content": str(prompt)}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if top_logprobs is not None:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
    return payload


def outer_command_line(
    *,
    label: str,
    candidate_env: Sequence[str] = (),
    n: int = len(LONG_PROMPT_SEEDS),
    port: int = DEFAULT_PORT,
    child_timeout_seconds: int = 5400,
) -> str:
    """The exact guarded outer command for one arm (also printed by --dry-run)."""

    child = [
        str(VENV_PYTHON),
        str(FABLE / "longprompt_agreement_screen.py"),
        "--label",
        str(label),
        "--n",
        str(int(n)),
        "--port",
        str(int(port)),
    ]
    for setting in candidate_env:
        child += ["--env", str(setting)]
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


def arm_command_lines(
    *, n: int = len(LONG_PROMPT_SEEDS), port: int = DEFAULT_PORT
) -> list[tuple[str, str]]:
    """``(label, command)`` for the control and the two prefill candidates."""

    return [
        (label, outer_command_line(label=label, candidate_env=env, n=n, port=port))
        for label, env in ARMS
    ]


def score_command_line(
    *, control: str = "control", candidate: str = "chunk4096"
) -> str:
    """The pure-python scoring command for two finished arms."""

    return shlex.join(
        [
            str(VENV_PYTHON),
            str(FABLE / "longprompt_agreement_screen.py"),
            "--score",
            str(OUT_DIR / f"{control}.json"),
            str(OUT_DIR / f"{candidate}.json"),
        ]
    )


# --------------------------------------------------------------------------
# Pure prompt construction (unit-tested with a fake tokenizer)
# --------------------------------------------------------------------------


def chunk_boundaries(total_tokens: int, chunk_size: int) -> list[int]:
    """Absolute offsets at which a prefill of ``total_tokens`` is cut.

    Pure arithmetic, used by the tests to prove the screen's prompt sizes
    actually make the 2,048 and 4,096 layouts differ -- the claim the whole
    screen rests on.
    """

    total = max(0, int(total_tokens))
    width = max(1, int(chunk_size))
    return list(range(width, total, width))


def rotate_context(context: str, seed: int) -> str:
    """Rotate the pinned coding context by ``seed`` lines.

    Deterministic and content-preserving: every seed sees the same bytes in a
    different order, so the six long prompts are genuinely distinct without
    introducing material the fixture hash does not cover.
    """

    lines = context.splitlines()
    if not lines:
        raise ValueError("coding context is empty")
    offset = int(seed) % len(lines)
    return "\n".join(lines[offset:] + lines[:offset])


def build_prompt_text(
    *,
    context: str,
    instruction: str,
    seed: int,
    target_tokens: int,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    count_templated: Callable[[str], int],
    tolerance: int = PROMPT_TOKEN_TOLERANCE,
    max_rounds: int = 6,
) -> dict[str, Any]:
    """A rotated slice of the pinned context, sized to ``target_tokens``.

    ``count_templated`` measures the FULL chat-templated prompt (what the
    server will actually prefill), so the target is the number the prefill
    chunker sees -- not the number of tokens in the user text. The loop
    corrects the context budget by the measured error instead of doing
    sentinel arithmetic on the template, which keeps it correct if the
    template changes.
    """

    if int(target_tokens) <= 0:
        raise ValueError("target_tokens must be positive")
    rotated = rotate_context(context, seed)
    body = str(instruction).strip() + INSTRUCTION_SUFFIX
    context_ids = list(encode(rotated.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("coding context encoded to zero tokens")

    def assemble(budget: int) -> str:
        take = max(1, int(budget))
        repeats = (take + len(context_ids) - 1) // len(context_ids)
        ids = (context_ids * repeats)[:take]
        return decode(ids).rstrip() + "\n\n" + body

    # First guess: the whole target, minus the body and the template, both of
    # which the correction loop measures for real on the next round. The loop
    # keeps the BEST attempt rather than the last: a one-token budget step can
    # be worth two templated tokens, so the sequence can oscillate around the
    # target without ever landing on it.
    budget = max(1, int(target_tokens) - len(list(encode(body))))
    history: list[dict[str, int]] = []
    best: tuple[int, str, int, int] | None = None
    text = assemble(budget)
    for _ in range(int(max_rounds)):
        measured = int(count_templated(text))
        history.append({"budget": int(budget), "templated_tokens": measured})
        error = int(target_tokens) - measured
        if best is None or abs(error) < best[0]:
            best = (abs(error), text, measured, int(budget))
        if error == 0:
            break
        budget = max(1, budget + error)
        text = assemble(budget)
    assert best is not None
    drift, text, measured, budget = best
    if drift > int(tolerance):
        raise RuntimeError(
            f"prompt sizing did not converge for seed {seed} target "
            f"{target_tokens} (best miss {drift} > tolerance {tolerance}): "
            f"{history}"
        )
    return {
        "seed": int(seed),
        "target_tokens": int(target_tokens),
        "templated_tokens": measured,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "context_budget_tokens": int(budget),
        "rounds": history,
    }


def prompt_plan(
    n_long: int, *, include_short: bool = True
) -> list[dict[str, Any]]:
    """``(name, seed, target_tokens)`` for the prompts one run will send."""

    if not 1 <= int(n_long) <= len(LONG_PROMPT_SEEDS):
        raise ValueError(
            f"--n must be between 1 and {len(LONG_PROMPT_SEEDS)}, got {n_long}"
        )
    plan = [
        {
            "name": f"long-16k-s{seed}",
            "seed": int(seed),
            "target_tokens": LONG_PROMPT_TOKENS,
        }
        for seed in LONG_PROMPT_SEEDS[: int(n_long)]
    ]
    if include_short:
        plan += [
            {"name": name, "seed": int(seed), "target_tokens": int(target)}
            for name, seed, target in SHORT_PROMPTS
        ]
    return plan


def load_tokenizer(model: Path) -> Any:
    """The pack's own tokenizer, CPU only. No MLX, no model weights."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)


def templated_ids(result: Any) -> list[int]:
    """Normalise ``apply_chat_template(tokenize=True)`` across transformers versions.

    transformers 5.x returns a ``BatchEncoding`` (so ``len()`` is the number of
    KEYS -- 2 -- not the number of tokens, which silently turned every prompt
    into a 2-token one), 4.x returns a flat list, and both can return a batch
    of one. Normalise all three to the flat id list.
    """

    if isinstance(result, Mapping):
        result = result["input_ids"]
    values = list(result)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise RuntimeError(
                f"chat template returned {len(values)} sequences, expected 1"
            )
        values = list(values[0])
    return [int(value) for value in values]


def tokenizer_callables(tokenizer: Any) -> dict[str, Callable[..., Any]]:
    """Adapt a HF tokenizer to the three callables ``build_prompt_text`` needs."""

    def count_templated(text: str) -> int:
        return len(
            templated_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        )

    return {
        "encode": lambda text: list(tokenizer.encode(text)),
        "decode": lambda ids: str(tokenizer.decode(list(ids))),
        "count_templated": count_templated,
    }


def build_prompts(
    *,
    plan: Sequence[Mapping[str, Any]],
    context: str,
    instruction: str,
    calls: Mapping[str, Callable[..., Any]],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for entry in plan:
        built = build_prompt_text(
            context=context,
            instruction=instruction,
            seed=int(entry["seed"]),
            target_tokens=int(entry["target_tokens"]),
            encode=calls["encode"],
            decode=calls["decode"],
            count_templated=calls["count_templated"],
        )
        built["name"] = str(entry["name"])
        prompts.append(built)
    return prompts


# --------------------------------------------------------------------------
# logprobs probe
# --------------------------------------------------------------------------


def probe_logprobs_support(
    base_url: str,
    *,
    model_id: str,
    post: Callable[..., Any],
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> dict[str, Any]:
    """One tiny request that asks whether this build accepts chat logprobs.

    Cheap, and self-correcting: today ``/v1/chat/completions`` answers 400
    ("support is planned"), and the day it does not, this screen starts
    recording top-K logprobs without an edit.
    """

    payload = build_chat_payload(
        "Say READY.",
        model_id=model_id,
        max_tokens=8,
        top_logprobs=int(top_logprobs),
    )
    try:
        response = post(f"{base_url}/v1/chat/completions", payload, timeout=180.0)
    except Exception as exc:  # HTTPError, URLError, timeouts
        detail = getattr(exc, "reason", None) or repr(exc)
        body = ""
        read = getattr(exc, "read", None)
        if callable(read):
            try:
                body = read().decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
        return {
            "supported": False,
            "top_logprobs": None,
            "detail": f"{detail}{(': ' + body) if body else ''}",
        }
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return {
            "supported": False,
            "top_logprobs": None,
            "detail": "server accepted logprobs but returned no logprobs.content",
        }
    return {"supported": True, "top_logprobs": int(top_logprobs), "detail": "ok"}


def extract_logprobs(choice: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """OpenAI ``choices[].logprobs.content`` -> the rows the scorer reads."""

    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return None
    rows: list[dict[str, Any]] = []
    for entry in content:
        top = [
            {"token": str(item.get("token")), "logprob": float(item.get("logprob"))}
            for item in (entry.get("top_logprobs") or [])
            if item.get("logprob") is not None
        ]
        top.sort(key=lambda item: -item["logprob"])
        rows.append(
            {
                "token": str(entry.get("token")),
                "logprob": (
                    None
                    if entry.get("logprob") is None
                    else float(entry["logprob"])
                ),
                "top": top,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Scoring (pure python -- no server, no GPU, no MLX)
# --------------------------------------------------------------------------


def exact_prefix_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Number of leading positions that match exactly."""

    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def top2_margin(row: Mapping[str, Any] | None) -> float | None:
    """Control's top-1 minus top-2 logprob at one position, in nats.

    A small margin is the rounding-class signature: two candidates the model
    could not separate, so which one wins is decided by the last bit of the
    reduction and a divergence there says nothing about quality.
    """

    if not row:
        return None
    top = list(row.get("top") or [])
    if len(top) < 2:
        return None
    values = sorted((float(item["logprob"]) for item in top), reverse=True)
    return float(values[0] - values[1])


def top1_logprob(row: Mapping[str, Any] | None) -> float | None:
    if not row:
        return None
    if row.get("logprob") is not None:
        return float(row["logprob"])
    top = list(row.get("top") or [])
    if not top:
        return None
    return float(max(float(item["logprob"]) for item in top))


def result_tokens(row: Mapping[str, Any]) -> list[Any]:
    """The comparison sequence for one prompt.

    Server-emitted logprob tokens first (exactly what the model produced),
    then the locally re-encoded ids, and text as the last resort. The receipt
    records which one was used via ``token_source``.
    """

    logprobs = row.get("logprobs")
    if logprobs:
        return [str(entry.get("token")) for entry in logprobs]
    ids = row.get("token_ids")
    if ids:
        return [int(value) for value in ids]
    return list(str(row.get("text") or ""))


def compare_prompt(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    near_tie_nats: float = DEFAULT_NEAR_TIE_NATS,
) -> dict[str, Any]:
    """One prompt's agreement row."""

    name = str(control.get("name"))
    if str(candidate.get("name")) != name:
        raise ValueError(
            f"receipt rows are not aligned: {name!r} vs {candidate.get('name')!r}"
        )
    if control.get("prompt_sha256") != candidate.get("prompt_sha256"):
        raise ValueError(
            f"{name}: the two arms were sent DIFFERENT prompts "
            f"({control.get('prompt_sha256')} vs {candidate.get('prompt_sha256')}) "
            "-- the comparison is void"
        )
    left = result_tokens(control)
    right = result_tokens(candidate)
    prefix = exact_prefix_length(left, right)
    identical = left == right

    control_lp = list(control.get("logprobs") or [])
    candidate_lp = list(candidate.get("logprobs") or [])
    has_logprobs = bool(control_lp and candidate_lp)
    deltas: list[float] = []
    for index in range(min(prefix, len(control_lp), len(candidate_lp))):
        a = top1_logprob(control_lp[index])
        b = top1_logprob(candidate_lp[index])
        if a is not None and b is not None:
            deltas.append(abs(a - b))

    margin: float | None = None
    near_tie: bool | None = None
    if not identical and prefix < len(control_lp):
        margin = top2_margin(control_lp[prefix])
        if margin is not None:
            near_tie = margin < float(near_tie_nats)

    return {
        "name": name,
        "prompt_tokens": control.get("prompt_tokens"),
        "target_tokens": control.get("target_tokens"),
        "control_length": len(left),
        "candidate_length": len(right),
        "prefix_length": prefix,
        "identical": identical,
        "token_source": control.get("token_source"),
        "has_logprobs": has_logprobs,
        "divergence_position": None if identical else prefix,
        "divergence_top2_margin_nats": margin,
        "divergence_is_near_tie": near_tie,
        "mean_top1_logprob_abs_delta": (
            statistics.fmean(deltas) if deltas else None
        ),
        "logprob_positions_compared": len(deltas),
        "text_divergence": None if identical else text_divergence(control, candidate),
    }


def text_divergence(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    before: int = 160,
    after: int = 80,
) -> dict[str, Any]:
    """Where the two completions part company, in CHARACTERS.

    The scorer never has a tokenizer, so this is what a human actually reads
    when a prompt is FLAGged: the shared tail, then each arm's next few
    characters. Independent of ``token_source``, so it is present whether the
    tokens came from the server or from a local re-encode.
    """

    left = str(control.get("text") or "")
    right = str(candidate.get("text") or "")
    index = exact_prefix_length(left, right)
    return {
        "char_prefix_length": index,
        "shared_tail": left[max(0, index - before) : index],
        "control_next": left[index : index + after],
        "candidate_next": right[index : index + after],
    }


def summarize_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefixes = [int(row["prefix_length"]) for row in rows]
    deltas = [
        float(row["mean_top1_logprob_abs_delta"])
        for row in rows
        if row.get("mean_top1_logprob_abs_delta") is not None
    ]
    full = [row for row in rows if row["identical"]]
    have = [row for row in rows if row.get("has_logprobs")]
    return {
        "prompts": len(rows),
        "median_prefix_length": (
            statistics.median(prefixes) if prefixes else None
        ),
        "min_prefix_length": min(prefixes) if prefixes else None,
        "max_prefix_length": max(prefixes) if prefixes else None,
        "full_agreement_prompts": len(full),
        "full_agreement_fraction": (len(full) / len(rows)) if rows else 0.0,
        "mean_top1_logprob_abs_delta": (
            statistics.fmean(deltas) if deltas else None
        ),
        # "available" is a property of the RECEIPTS, not of whether a
        # comparison happened to produce numbers: with logprobs present but
        # every prompt diverging at position 0 there are no deltas to average,
        # and calling that "unavailable" would quietly swap in the strict rule.
        "logprob_evidence": (
            "available"
            if have and len(have) == len(rows)
            else ("partial" if have else "unavailable")
        ),
        "prompts_with_logprobs": len(have),
    }


def verdict(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    near_tie_nats: float = DEFAULT_NEAR_TIE_NATS,
    max_logprob_delta: float = DEFAULT_MAX_LOGPROB_DELTA,
) -> dict[str, Any]:
    """PASS / FLAG. The rule is stated in ``--help`` and in the README.

    With logprobs: PASS when every divergence is at a near-tie (control top-2
    margin < ``near_tie_nats``) AND the mean top-1 logprob |delta| over the
    agreeing prefixes is < ``max_logprob_delta``.

    Without logprobs: PASS only on full agreement everywhere. A divergence
    with no margin behind it cannot be classed as a rounding tie, so it is
    FLAGged rather than excused.
    """

    reasons: list[str] = []
    offenders: list[dict[str, Any]] = []

    for row in rows:
        if row["identical"]:
            continue
        near_tie = row.get("divergence_is_near_tie")
        if near_tie is True:
            continue
        offenders.append(
            {
                "name": row["name"],
                "position": row["divergence_position"],
                "prefix_length": row["prefix_length"],
                "top2_margin_nats": row["divergence_top2_margin_nats"],
                "text_divergence": row.get("text_divergence"),
            }
        )
        if near_tie is False:
            reasons.append(
                f"{row['name']}: diverged at position {row['divergence_position']} "
                f"where the control's top-2 margin was "
                f"{row['divergence_top2_margin_nats']:.4f} nats "
                f"(>= {near_tie_nats})"
            )
        else:
            reasons.append(
                f"{row['name']}: diverged at position "
                f"{row['divergence_position']} and this build supplies no "
                "logprobs, so the divergence cannot be shown to be a near-tie"
            )

    mean_delta = summary.get("mean_top1_logprob_abs_delta")
    if mean_delta is not None and float(mean_delta) >= float(max_logprob_delta):
        reasons.append(
            f"mean top-1 logprob |delta| over agreeing prefixes is "
            f"{float(mean_delta):.5f} (>= {max_logprob_delta})"
        )

    return {
        "verdict": "PASS" if not reasons else "FLAG",
        "reasons": reasons,
        "offenders": offenders,
        "logprob_evidence": summary.get("logprob_evidence"),
        "thresholds": {
            "near_tie_nats": float(near_tie_nats),
            "max_logprob_delta": float(max_logprob_delta),
        },
        "rule": (
            "PASS if every divergence is at a near-tie (control top-2 margin "
            f"< {near_tie_nats} nats) AND mean top-1 logprob |delta| over the "
            f"agreeing prefixes < {max_logprob_delta}; without logprobs, PASS "
            "requires full agreement on every prompt."
        ),
    }


def rows_by_name(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["name"]): row for row in receipt.get("results") or []}


def compare_receipts(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    near_tie_nats: float = DEFAULT_NEAR_TIE_NATS,
    max_logprob_delta: float = DEFAULT_MAX_LOGPROB_DELTA,
) -> dict[str, Any]:
    left = rows_by_name(control)
    right = rows_by_name(candidate)
    missing = sorted(set(left) ^ set(right))
    if missing:
        raise ValueError(
            f"the two receipts do not cover the same prompts: {missing}"
        )
    if not left:
        raise ValueError("receipts contain no results")
    rows = [
        compare_prompt(left[name], right[name], near_tie_nats=near_tie_nats)
        for name in sorted(left)
    ]
    summary = summarize_agreement(rows)
    return {
        "control_label": control.get("label"),
        "candidate_label": candidate.get("label"),
        "control_env": (control.get("flags") or {}).get("candidate_env"),
        "candidate_env": (candidate.get("flags") or {}).get("candidate_env"),
        "per_prompt": rows,
        "summary": summary,
        "verdict": verdict(
            rows,
            summary,
            near_tie_nats=near_tie_nats,
            max_logprob_delta=max_logprob_delta,
        ),
    }


def noise_floor(
    control_a: Mapping[str, Any],
    control_b: Mapping[str, Any],
    *,
    near_tie_nats: float = DEFAULT_NEAR_TIE_NATS,
) -> dict[str, Any]:
    """Control vs control: the screen's own run-to-run agreement.

    Greedy decoding is deterministic, so this SHOULD be 100%. Anything else
    means the lane is nondeterministic and no candidate comparison run
    against it means anything.
    """

    left = rows_by_name(control_a)
    right = rows_by_name(control_b)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("the two control receipts share no prompts")
    rows = [
        compare_prompt(left[name], right[name], near_tie_nats=near_tie_nats)
        for name in shared
    ]
    summary = summarize_agreement(rows)
    summary["deterministic"] = all(row["identical"] for row in rows)
    return {
        "labels": [control_a.get("label"), control_b.get("label")],
        "per_prompt": rows,
        "summary": summary,
    }


def render_report(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    summary = report["summary"]
    lines.append(
        f"[longprompt-screen] {report.get('control_label')} vs "
        f"{report.get('candidate_label')}  "
        f"env={json.dumps(report.get('candidate_env') or {}, sort_keys=True)}"
    )
    for row in report["per_prompt"]:
        state = "IDENTICAL" if row["identical"] else "DIVERGED"
        detail = ""
        if not row["identical"]:
            margin = row["divergence_top2_margin_nats"]
            detail = (
                f" at {row['divergence_position']}"
                + (
                    f", control top-2 margin {margin:.4f} nats"
                    f" ({'near-tie' if row['divergence_is_near_tie'] else 'confident'})"
                    if margin is not None
                    else ", no margin available"
                )
            )
        lines.append(
            f"  {row['name']:<22} prompt={row['prompt_tokens']!s:>6} "
            f"prefix={row['prefix_length']}/{row['control_length']} {state}{detail}"
        )
    lines.append(
        f"  median prefix {summary['median_prefix_length']}, min "
        f"{summary['min_prefix_length']}, full agreement "
        f"{summary['full_agreement_prompts']}/{summary['prompts']} "
        f"({summary['full_agreement_fraction']:.2f}), mean top-1 logprob "
        f"|delta| {summary['mean_top1_logprob_abs_delta']} "
        f"[logprobs {summary['logprob_evidence']}]"
    )
    lines.append(f"  VERDICT: {report['verdict']['verdict']}")
    for reason in report["verdict"]["reasons"]:
        lines.append(f"    - {reason}")
    return "\n".join(lines)


def render_noise_floor(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    head = (
        f"[longprompt-screen] control noise floor "
        f"({report['labels'][0]} vs {report['labels'][1]}): "
        f"full agreement {summary['full_agreement_prompts']}/{summary['prompts']}, "
        f"min prefix {summary['min_prefix_length']}"
    )
    if summary.get("deterministic"):
        return head + " -- deterministic"
    return (
        head
        + " -- NOT deterministic; the candidate comparison above is VOID until "
        "this is explained"
    )


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def arm_identity(
    candidate_env: Mapping[str, str],
    family_env: Mapping[str, str],
    server_argv: Sequence[str],
    prompts: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Everything that must match for a resumed run to still be ONE arm."""

    return {
        "candidate_env": dict(candidate_env),
        "family_env": dict(family_env),
        "server_argv": list(server_argv),
        "max_tokens": int(args.max_tokens),
        "model_id": str(args.model_id),
        "prompts": [
            {
                "name": str(prompt["name"]),
                "seed": int(prompt["seed"]),
                "target_tokens": int(prompt["target_tokens"]),
                "text_sha256": str(prompt["text_sha256"]),
            }
            for prompt in prompts
        ],
    }


def existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["name"])] = row
    return rows


def generate_results(
    *,
    base_url: str,
    prompts: Sequence[Mapping[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
    screen: Any,
    logprobs: Mapping[str, Any],
    encode: Callable[[str], Sequence[int]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_path = out_dir / "results.jsonl"
    done = existing_results(results_path) if args.resume else {}
    started = time.time()
    generated = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    finish_reasons: dict[str, int] = {}
    top_logprobs = logprobs.get("top_logprobs") if logprobs.get("supported") else None

    with results_path.open("a", encoding="utf-8") as handle:
        for prompt in prompts:
            name = str(prompt["name"])
            if name in done:
                continue
            payload = build_chat_payload(
                str(prompt["text"]),
                model_id=args.model_id,
                max_tokens=args.max_tokens,
                top_logprobs=top_logprobs,
            )
            request_id = f"longprompt-screen-{args.label}-{name}"
            request_started = time.perf_counter()
            response = screen.http_post(
                f"{base_url}/v1/chat/completions",
                payload,
                timeout=args.request_timeout,
                headers={"x-mtplx-request-id": request_id},
            )
            elapsed = time.perf_counter() - request_started
            choice = response["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"non-text completion for {name}: {content!r}")
            if choice["message"].get("reasoning_content"):
                raise RuntimeError(
                    f"{name}: server emitted reasoning with thinking off "
                    "-- the screen contract is broken, not the kernel"
                )
            usage = response.get("usage", {}) or {}
            rows = extract_logprobs(choice)
            token_ids: list[int] | None = None
            token_source = "text"
            if rows:
                token_source = "server_logprobs"
            elif encode is not None:
                token_ids = [int(value) for value in encode(content)]
                token_source = "local_reencode"
            row = {
                "name": name,
                "seed": int(prompt["seed"]),
                "target_tokens": int(prompt["target_tokens"]),
                "prompt_sha256": str(prompt["text_sha256"]),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "finish_reason": choice.get("finish_reason"),
                "text": content,
                "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "token_ids": token_ids,
                "token_source": token_source,
                "logprobs": rows,
                "elapsed_s": elapsed,
                "response_id": response.get("id"),
                "request_id": request_id,
                "usage": usage,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[name] = row
            generated += 1
            finish = str(choice.get("finish_reason"))
            print(
                f"[longprompt-screen:{args.label}] {len(done)}/{len(prompts)} "
                f"{name} prompt={row['prompt_tokens']} "
                f"completion={row['completion_tokens']} {elapsed:.2f}s "
                f"finish={finish} sha={row['text_sha256'][:12]}",
                flush=True,
            )

    ordered = [done[str(prompt["name"])] for prompt in prompts]
    # Totalled over EVERY row in the arm, not just the ones this process
    # generated: a resumed run's receipt otherwise under-reports its own work.
    for row in ordered:
        for key in usage_total:
            usage_total[key] += int((row.get("usage") or {}).get(key, 0) or 0)
        finish = str(row.get("finish_reason"))
        finish_reasons[finish] = finish_reasons.get(finish, 0) + 1
    return ordered, {
        "results": str(results_path),
        "generated": generated,
        "resumed": len(ordered) - generated,
        "wall_s": time.time() - started,
        "usage_total": usage_total,
        "finish_reasons": finish_reasons,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Candidate MTPLX_* env exported before the server starts "
            "(repeatable). No --env at all IS the control arm. "
            "MTPLX_PREFILL_CHUNK_SIZE must be set together with "
            "MTPLX_QSA_PREFILL_COMPILE_ROWS."
        ),
    )
    parser.add_argument("--label", help="Receipt name. Required unless --score.")
    parser.add_argument(
        "--n",
        type=int,
        default=len(LONG_PROMPT_SEEDS),
        help=(
            f"Number of ~16K prompts, 1..{len(LONG_PROMPT_SEEDS)} "
            f"(seeds {LONG_PROMPT_SEEDS[0]}..{LONG_PROMPT_SEEDS[-1]}). Default "
            f"{len(LONG_PROMPT_SEEDS)}."
        ),
    )
    parser.add_argument(
        "--short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also send the 9,216 and 4,608-token prompts, whose final chunk "
            "differs between the 2,048 and 4,096 layouts. On by default."
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--receipt-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--python", type=Path, default=VENV_PYTHON)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--server-ready-timeout", type=float, default=1200.0)
    parser.add_argument("--warmup-timeout", type=float, default=900.0)
    parser.add_argument(
        "--guard-mode",
        choices=("auto", "attestation", "window"),
        default="auto",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the prompts (tokenizer only, no GPU), print the outer "
            "guarded command and the server argv, then exit."
        ),
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the three guarded arm commands and the scoring command, then exit.",
    )
    parser.add_argument(
        "--score",
        nargs="+",
        metavar="RECEIPT",
        help=(
            "Score finished receipts, no GPU: CONTROL CANDIDATE "
            "[SECOND_CONTROL]. The third receipt is reported as the "
            "run-to-run noise floor. VERDICT: PASS if every divergence is at "
            f"a near-tie (control top-2 margin < {DEFAULT_NEAR_TIE_NATS} nats) "
            "AND the mean top-1 logprob |delta| over agreeing prefixes is "
            f"< {DEFAULT_MAX_LOGPROB_DELTA}; when the receipts carry no "
            "logprobs (this build rejects them on /v1/chat/completions), PASS "
            "requires full agreement on every prompt and any divergence is "
            "FLAGged for human review."
        ),
    )
    parser.add_argument(
        "--near-tie-nats", type=float, default=DEFAULT_NEAR_TIE_NATS
    )
    parser.add_argument(
        "--max-logprob-delta", type=float, default=DEFAULT_MAX_LOGPROB_DELTA
    )
    parser.add_argument(
        "--json", action="store_true", help="--score: emit the full JSON report."
    )
    return parser


def run_score(args: argparse.Namespace) -> int:
    paths = [Path(value) for value in args.score]
    if not 2 <= len(paths) <= 3:
        raise SystemExit(
            "--score takes CONTROL CANDIDATE [SECOND_CONTROL], got "
            f"{len(paths)} paths"
        )
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    report = compare_receipts(
        receipts[0],
        receipts[1],
        near_tie_nats=args.near_tie_nats,
        max_logprob_delta=args.max_logprob_delta,
    )
    print(render_report(report))
    if len(receipts) == 3:
        floor = noise_floor(
            receipts[0], receipts[2], near_tie_nats=args.near_tie_nats
        )
        report["noise_floor"] = floor
        print(render_noise_floor(floor))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"]["verdict"] == "PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.print_commands:
        prompt_plan(args.n, include_short=args.short)  # validate --n before printing
        for label, command in arm_command_lines(n=args.n, port=args.port):
            print(f"# {label}")
            print(command)
            print()
        print("# score (no GPU)")
        print(score_command_line())
        return 0

    if args.score:
        return run_score(args)

    if not args.label:
        raise SystemExit("--label is required (or use --score / --print-commands)")

    screen = load_screen()
    if args.model is None:
        args.model = screen.MODEL
    if args.model_id is None:
        args.model_id = screen.MODEL_ID

    candidate_env = screen.parse_env_settings(args.env)
    assert_candidate_env_coherent(candidate_env)
    family_env = build_screen_family_env(screen.CONTROL_FAMILY_ENV)
    server_argv = build_server_argv(
        python=args.python,
        model=args.model,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
    )

    driver = screen.load_abba_driver()
    # The driver's own SHA-256 pin on the fixture pair. Calling it is the
    # drift gate: if the fixtures moved, this raises before anything else.
    driver.production_prompt_content()
    context = (driver.FIXTURES / "qwen38_generation_context.py").read_text()
    instruction = json.loads(
        (driver.FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
        .read_text()
        .splitlines()[0]
    )["prompt"]

    tokenizer = load_tokenizer(Path(args.model))
    calls = tokenizer_callables(tokenizer)
    plan = prompt_plan(args.n, include_short=args.short)
    prompts = build_prompts(
        plan=plan, context=context, instruction=instruction, calls=calls
    )
    for prompt in prompts:
        print(
            f"[longprompt-screen] prompt {prompt['name']}: "
            f"{prompt['templated_tokens']} templated tokens "
            f"(target {prompt['target_tokens']}), "
            f"sha={prompt['text_sha256'][:12]}",
            flush=True,
        )

    if args.dry_run:
        print("[longprompt-screen] outer command:")
        print("  " + outer_command_line(
            label=args.label, candidate_env=args.env, n=args.n, port=args.port
        ))
        print("[longprompt-screen] server argv:")
        print("  " + " ".join(server_argv))
        print("[longprompt-screen] candidate env: " + json.dumps(candidate_env))
        print("[longprompt-screen] family env: " + json.dumps(family_env, sort_keys=True))
        return 0

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
        f"[longprompt-screen] lock attested ({guard['mode']}); "
        f"reclaimable={available / 1024**3:.2f} GiB",
        flush=True,
    )

    run_dir = args.receipt_dir.resolve() / str(args.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    receipt_path = run_dir.parent / f"{args.label}.json"
    arm_claim = screen.claim_arm_identity(
        run_dir,
        arm_identity(candidate_env, family_env, server_argv, prompts, args),
    )
    print(f"[longprompt-screen] arm identity {arm_claim['claimed']}", flush=True)

    environment = screen.build_server_env(
        os.environ, candidate_env, family=family_env
    )
    base_url = f"http://{args.host}:{args.port}"
    timings: dict[str, float] = {}
    run_started = time.time()
    arm = "candidate" if candidate_env else "control"
    print(
        f"[longprompt-screen] arm={arm} "
        f"env={json.dumps(candidate_env, sort_keys=True)}",
        flush=True,
    )
    print("[longprompt-screen] " + " ".join(server_argv), flush=True)

    process: subprocess.Popen[Any] | None = None
    stop_receipt: dict[str, Any] = {}
    warmup: dict[str, Any] = {}
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
            settings = screen.http_get(
                f"{base_url}/v1/mtplx/settings", timeout=15.0
            )
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
                print(
                    "[longprompt-screen] no warmup.background in /health; "
                    "not waiting",
                    flush=True,
                )

            ready = screen.http_post(
                f"{base_url}/v1/chat/completions",
                build_chat_payload(
                    "Reply with the single word READY.",
                    model_id=args.model_id,
                    max_tokens=16,
                ),
                timeout=300.0,
            )
            ready_text = ready["choices"][0]["message"].get("content")
            if not isinstance(ready_text, str) or not ready_text.strip():
                raise RuntimeError(f"READY chat returned no content: {ready!r}")
            timings["server_ready_s"] = time.time() - server_started
            print(
                f"[longprompt-screen] ready in {timings['server_ready_s']:.0f}s",
                flush=True,
            )

            logprobs = probe_logprobs_support(
                base_url, model_id=args.model_id, post=screen.http_post
            )
            print(
                "[longprompt-screen] chat logprobs "
                f"{'SUPPORTED' if logprobs['supported'] else 'UNSUPPORTED'}: "
                f"{logprobs['detail']}",
                flush=True,
            )

            results, generation = generate_results(
                base_url=base_url,
                prompts=prompts,
                out_dir=run_dir,
                args=args,
                screen=screen,
                logprobs=logprobs,
                encode=calls["encode"],
            )
            timings["generation_s"] = generation["wall_s"]
    finally:
        if process is not None:
            stop_receipt = screen.stop_server(process)
            print(f"[longprompt-screen] server stopped: {stop_receipt}", flush=True)

    timings["total_s"] = time.time() - run_started
    receipt = {
        "schema": "mtplx-fable-longprompt-agreement-v1",
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm,
        "flags": {
            "candidate_env": dict(candidate_env),
            "family_env": dict(family_env),
            "memory_env": dict(MEMORY_ENV),
            "never_exported": dict(screen.NEVER_EXPORT),
            "server_argv": list(server_argv),
        },
        "model": dict(provenance),
        "guard": dict(guard),
        "sampler": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(args.max_tokens),
            "n": 1,
            "greedy": True,
            "enable_thinking": False,
        },
        "logprobs": dict(logprobs),
        "prompts": [
            {
                key: prompt[key]
                for key in (
                    "name",
                    "seed",
                    "target_tokens",
                    "templated_tokens",
                    "text_sha256",
                    "chars",
                    "context_budget_tokens",
                )
            }
            for prompt in prompts
        ],
        "results": results,
        "generation": dict(generation),
        "server_health": dict(health),
        "server_settings": dict(settings),
        "server_warmup": dict(warmup),
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
    print(
        f"[longprompt-screen] {args.label}: {len(results)} prompts, "
        f"{generation['usage_total']['prompt_tokens']} prompt tokens, "
        f"{generation['usage_total']['completion_tokens']} completion tokens",
        flush=True,
    )
    print(f"wrote {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
