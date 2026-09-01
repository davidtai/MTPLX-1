#!/usr/bin/env python3
"""HumanEval pass@1 quality screen for rounding-class Qwen3.8 Flash-Next kernels.

Why this exists
---------------
``scripts/fable/abba_window.py`` measures SPEED. Candidates whose numerics are
only a rounding class away from the eager chain -- ``MTPLX_FABLE_HC_M4=1`` is
the first one -- cannot be gated on an output digest, because they are not
bit-exact and never claimed to be (see the numerics section of
``mtplx/kernels/qwen4_m4_hyper_read.py``). David's rule for that class is a
TASK eval: the full HumanEval (164 problems) pass@1 must look decent and sit
within noise of the control. ``--n 20`` is a smoke, never a verdict.

Shape of one run
----------------
1. consume the canonical GPU guard evidence (shared with ``abba_driver``),
2. wait for reclaimable memory, then start ONE MTPLX server on :8091 from this
   worktree's venv with the control family env plus ``--env`` candidates,
3. wait for ``/health`` + background warmup + a READY chat,
4. generate one greedy completion per HumanEval problem (EvalPlus 0.3.1's own
   OpenAI prompt and sanitizer),
5. stop the server, THEN score with the ``evalplus.evaluate`` CLI in the
   evalplus checkout's own venv (CPU only, no model resident),
6. write a receipt under ``.benchmark-artifacts/fable/evals/``.

Run it THROUGH ``bench/laguna/run_guarded.py`` -- see ``scripts/fable/README.md``
for the two exact outer command lines (control and candidate).

Family environment
------------------
The server already family-defaults most of the control lane itself
(``mtplx/server/openai.py:_server_runtime_env_overrides``: AR_PIPELINE,
COMPILED_GDN, FAMILY_CAPTURE_COMMIT, FUSED_HC_V3, the GDN/GLU fusions,
QSA_GATHER, batched target distributions, and -- for a fixed-M4 pack --
COMPILED_VERIFY / QWEN4_FIXED_M4_VERIFY / M4_STAGE3 / QSA_M4_FUSED_KV_GATHER).
``CONTROL_FAMILY_ENV`` below states all of those explicitly plus the four the
server does NOT default, so the receipt records the lane instead of implying
it. Every key in it is either absent from the turbo profile or a member of
``PROFILE_ENV_USER_OVERRIDE_KEYS``, so ``apply_profile_env`` cannot stomp it.

``MTPLX_NAX_VERIFY`` is deliberately NOT exported -- see ``NEVER_EXPORT``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FABLE = ROOT / "scripts" / "fable"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

MODEL = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
MODEL_ID = "mtplx-flash-next-optimized-speed"
EXPECTED_MODEL_SHA = "29ba90f82124961d0d902a9ea9bbb1034972af2f"

OUT_DIR = ROOT / ".benchmark-artifacts" / "fable" / "evals"
RUN_GUARDED = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py"
)
QWEN_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")

EVALPLUS_ROOT = Path("/Users/davidtai/projects/evalplus")
EVALPLUS_EVALUATE = EVALPLUS_ROOT / ".venv" / "bin" / "evalplus.evaluate"
EVALPLUS_SITE = EVALPLUS_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
EVALPLUS_VERSION = "0.3.1"
HUMANEVAL_TASK_COUNT = 164

# EvalPlus 0.3.1's own OpenAI chat prompt, byte-identical to the one
# bench/qwen35b-mtp-b8-evalplus-20260809/evalplus_paired_codegen.py used and to
# evalplus.provider.utility.make_raw_chat_prompt.
SYSTEM_MESSAGE = "You are a helpful assistant good at coding."
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)

DEFAULT_PORT = 8091
DEFAULT_MAX_TOKENS = 768
DEFAULT_HOST = "127.0.0.1"

#: The control lane, stated in full. See the module docstring.
CONTROL_FAMILY_ENV: dict[str, str] = {
    # --- served by the server's own qwen4_exp family defaults -------------
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "0",
    "MTPLX_AR_PIPELINE": "1",
    "MTPLX_COMPILED_GDN": "1",
    "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
    "MTPLX_FUSED_HC_V3": "1",
    "MTPLX_FUSED_GDN_INPROJ": "1",
    "MTPLX_FUSED_GATE_UP": "1",
    "MTPLX_FUSED_GDN_CONVNORM": "1",
    "MTPLX_FUSED_GDN_STEP": "1",
    "MTPLX_FUSED_CONVNORM_VERIFY": "1",
    "MTPLX_QSA_GATHER": "1",
    "MTPLX_BATCH_TARGET_ARRAYS": "1",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "0",
    "MTPLX_COMPILED_VERIFY": "1",
    "MTPLX_QWEN4_FIXED_M4_VERIFY": "1",
    "MTPLX_QWEN4_M4_STAGE3": "1",
    "MTPLX_QSA_M4_FUSED_KV_GATHER": "1",
    # --- the ABBA control arm's additions the server does NOT default -----
    # abba_window.CONTROL_FLAGS --compiled-mtp-prepare / --full-frspec, and
    # abba_window.CONTROL_CANDIDATE_ENV's three routed-down/GLU keys.
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "1",
    "MTPLX_FRSPEC_DRAFT": "1",
    "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE": "1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL": "1",
    "MTPLX_QWEN4_M4_ROUTED_GLU": "1",
}

#: Keys that must reach the server UNSET, with the reason.
#:
#: MTPLX_NAX_VERIFY is the trap. The turbo profile sets it to "1" and it is not
#: a member of PROFILE_ENV_USER_OVERRIDE_KEYS, so an exported "0" is stomped
#: back to "1" by apply_profile_env. The server only reaches its own
#: overrides["MTPLX_NAX_VERIFY"] = "0" (applied AFTER the profile, so it wins)
#: when the launcher left the key unset. Exporting the value we want here would
#: therefore produce the value we do not want.
NEVER_EXPORT: dict[str, str] = {
    "MTPLX_NAX_VERIFY": (
        "turbo sets 1 and the key is not operator-overridable; leaving it "
        "unset is what lets the server's own family override set it to 0"
    ),
}


# --------------------------------------------------------------------------
# Guard (shared with abba_driver -- one implementation, not a copy)
# --------------------------------------------------------------------------


def _load_sibling(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_abba_driver() -> Any:
    """abba_driver is import-safe: every MLX import in it is deferred."""

    return _load_sibling("_fable_abba_driver", FABLE / "abba_driver.py")


def load_run_guarded() -> Any:
    """run_guarded is import-safe too (mtplx.qwen_guard is imported in main)."""

    return _load_sibling("_fable_run_guarded", RUN_GUARDED)


# --------------------------------------------------------------------------
# Pure argv / environment construction (unit-tested, no GPU, no network)
# --------------------------------------------------------------------------


def parse_env_settings(settings: Sequence[str]) -> dict[str, str]:
    """Parse repeatable ``--env KEY=VALUE`` into a dict, fail-closed."""

    parsed: dict[str, str] = {}
    for setting in settings:
        try:
            key, value = str(setting).split("=", 1)
        except ValueError as exc:
            raise ValueError(f"invalid --env value: {setting!r}") from exc
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"invalid --env value: {setting!r}")
        if not key.startswith("MTPLX_"):
            raise ValueError(f"--env keys must start with MTPLX_: {setting!r}")
        if key in parsed:
            raise ValueError(f"duplicate --env key: {key}")
        if key in NEVER_EXPORT:
            raise ValueError(f"{key} must not be exported: {NEVER_EXPORT[key]}")
        parsed[key] = value
    return parsed


def build_server_env(
    base: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    family: Mapping[str, str] = CONTROL_FAMILY_ENV,
) -> dict[str, str]:
    """The server's environment: base minus every MTPLX_*, plus family, plus candidate.

    Stripping the inherited MTPLX_* namespace is what makes two arms in the
    same shell comparable: a leftover export from a previous run would
    otherwise silently move the lane under one of them.
    """

    environment = {
        key: value
        for key, value in base.items()
        if not str(key).startswith("MTPLX_")
    }
    environment.update(family)
    for key, value in candidate.items():
        if key in NEVER_EXPORT:
            raise ValueError(f"{key} must not be exported: {NEVER_EXPORT[key]}")
        environment[key] = value
    environment["HF_HUB_OFFLINE"] = "1"
    for key in NEVER_EXPORT:
        if key in environment:
            raise RuntimeError(f"{key} leaked into the server environment")
    return environment


def build_server_argv(
    *,
    python: str | Path = VENV_PYTHON,
    model: str | Path = MODEL,
    model_id: str = MODEL_ID,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[str]:
    """The MTPLX OpenAI server command line for one screen arm.

    Mirrors the production launcher (start-qwen38-flash-next-210.sh: model,
    model-id, host, port, everything else defaulted) and then pins the three
    things a quality screen must not inherit from a default: reasoning OFF (so
    164 problems do not each burn a thousand thinking tokens), a greedy server
    sampler, and no cross-request SSD session cache.
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
        # --reasoning-mode off resolves to args.reasoning="off" which forces
        # args.enable_thinking=False (openai.py parse_args tail). Asserted
        # against /v1/mtplx/settings before a single problem is generated.
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


def build_chat_payload(
    prompt: str,
    *,
    model_id: str = MODEL_ID,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """EvalPlus 0.3.1's chat request, with thinking pinned off per request."""

    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    f"{INSTRUCTION_PREFIX}\n```python\n{str(prompt).strip()}\n```"
                ),
            },
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def select_task_ids(task_ids: Sequence[str], n: int) -> list[str]:
    """The first ``n`` problems in dataset order.

    ``n`` is 164 (the whole set, the only verdict-grade size) or 20 (a smoke).
    Dataset order is HumanEval/0.. so the smoke is reproducible without a seed.
    """

    ordered = list(task_ids)
    if len(ordered) != HUMANEVAL_TASK_COUNT:
        raise ValueError(
            f"expected {HUMANEVAL_TASK_COUNT} HumanEval problems, got {len(ordered)}"
        )
    if n not in (20, HUMANEVAL_TASK_COUNT):
        raise ValueError(f"--n must be 20 or {HUMANEVAL_TASK_COUNT}, got {n}")
    return ordered[:n]


def summarize_scores(
    eval_results: Mapping[str, Any], scored_task_ids: Sequence[str]
) -> dict[str, Any]:
    """pass@1 over exactly the scored subset, plus the per-problem pass list.

    HumanEval+ pass@1 is ``base_status == plus_status == pass``, NOT
    ``plus_status == pass``: the plus tests are the EXTRA inputs only, so a
    solution can pass them while failing a base test. Counting plus_status
    alone reported 149/164 for the 2026-08-24 native-MTP samples where
    evalplus's own CLI reports 148 -- see
    ``evalplus/evaluate.py`` (``new_correct``) and the validation in
    ``tests/test_fable_humaneval_screen.py``.

    ``evalplus.evaluate`` insists on a samples file that covers all 164
    problems, so a ``--n 20`` run pads the other 144 with an empty solution and
    this function then ignores them. Padding rows are never counted.
    """

    rows: list[dict[str, Any]] = []
    base_passed = 0
    plus_passed = 0
    for task_id in scored_task_ids:
        try:
            completions = eval_results["eval"][task_id]
        except KeyError as exc:
            raise KeyError(f"no eval result for {task_id}") from exc
        if len(completions) != 1:
            raise ValueError(
                f"{task_id}: expected exactly 1 completion, got {len(completions)}"
            )
        row = completions[0]
        base_ok = str(row.get("base_status")) == "pass"
        plus_ok = base_ok and str(row.get("plus_status")) == "pass"
        base_passed += int(base_ok)
        plus_passed += int(plus_ok)
        rows.append(
            {
                "task_id": task_id,
                "base_pass": base_ok,
                "plus_pass": plus_ok,
            }
        )
    total = len(rows)
    return {
        "tasks": total,
        "humaneval": {
            "passed": base_passed,
            "pass_at_1": (base_passed / total) if total else 0.0,
        },
        "humaneval_plus": {
            "passed": plus_passed,
            "pass_at_1": (plus_passed / total) if total else 0.0,
        },
        "per_problem": rows,
        "base_failures": [row["task_id"] for row in rows if not row["base_pass"]],
        "plus_failures": [row["task_id"] for row in rows if not row["plus_pass"]],
    }


def outer_command_line(
    *,
    label: str = "control",
    candidate_env: Sequence[str] = (),
    n: int = HUMANEVAL_TASK_COUNT,
    port: int = DEFAULT_PORT,
    child_timeout_seconds: int = 5400,
) -> str:
    """The exact guarded outer command for one arm (also printed by --dry-run)."""

    import shlex

    child = [
        str(VENV_PYTHON),
        str(FABLE / "humaneval_screen.py"),
        "--label",
        label,
        "--n",
        str(int(n)),
        "--port",
        str(int(port)),
    ]
    for setting in candidate_env:
        child += ["--env", setting]
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
# HTTP helpers
# --------------------------------------------------------------------------


def http_get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def http_post(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, int(port))) != 0


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-int(count) :]


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------


def wait_for_health(
    base_url: str,
    *,
    process: subprocess.Popen[Any],
    log_path: Path,
    timeout: float,
    poll: float = 5.0,
) -> dict[str, Any]:
    """Poll /health until the server answers ok, or the server dies."""

    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode} during startup; "
                f"log tail:\n" + "\n".join(tail_lines(log_path, 40))
            )
        try:
            health = http_get(f"{base_url}/health", timeout=5.0)
            if health.get("ok"):
                return health
            last_error = f"health not ok: {health!r}"
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"server not ready after {timeout:.0f}s ({last_error}); log tail:\n"
                + "\n".join(tail_lines(log_path, 40))
            )
        time.sleep(poll)


def assert_server_contract(health: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """Refuse to score a server that is not the configuration we asked for."""

    problems: list[str] = []

    def require(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, observed {actual!r}")

    require("health.model", health.get("model"), MODEL_ID)
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


def stop_server(
    process: subprocess.Popen[Any], *, grace_seconds: float = 180.0
) -> dict[str, Any]:
    """SIGTERM, wait, then SIGKILL. Never launchctl, never the flock."""

    if process.poll() is not None:
        return {"stopped": "already_exited", "returncode": process.returncode}
    started = time.monotonic()
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return {"stopped": "vanished", "returncode": process.poll()}
    try:
        process.wait(timeout=grace_seconds)
        return {
            "stopped": "sigterm",
            "returncode": process.returncode,
            "wait_s": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=60)
    return {
        "stopped": "sigkill",
        "returncode": process.returncode,
        "wait_s": time.monotonic() - started,
    }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def load_evalplus() -> tuple[Callable[[], dict[str, Any]], Callable[..., str]]:
    """Reuse the evalplus checkout in place (the bench harnesses' own pattern)."""

    site = str(EVALPLUS_SITE)
    if site not in sys.path:
        sys.path.append(site)
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize

    return get_human_eval_plus, sanitize


def arm_identity(
    candidate_env: Mapping[str, str],
    server_argv: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Everything that must match for a resumed run to still be ONE arm."""

    return {
        "candidate_env": dict(candidate_env),
        "control_family_env": dict(CONTROL_FAMILY_ENV),
        "server_argv": list(server_argv),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "model_id": str(args.model_id),
    }


def claim_arm_identity(run_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Write the arm identity, or refuse to append to a DIFFERENT arm.

    Resume is on by default, and ``--label`` is the only thing choosing the
    output directory. Without this, re-running a label with a different --env
    would silently append candidate completions to a control's samples.jsonl
    and score the mixture.
    """

    path = run_dir / "arm.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != dict(identity):
            drift = {
                key: (previous.get(key), dict(identity).get(key))
                for key in set(previous) | set(identity)
                if previous.get(key) != dict(identity).get(key)
            }
            raise RuntimeError(
                f"{path} already describes a DIFFERENT arm; use a new --label "
                f"or delete {run_dir}. Drift: {json.dumps(drift, sort_keys=True)}"
            )
        return {"claimed": "resumed", "path": str(path)}
    path.write_text(
        json.dumps(dict(identity), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"claimed": "new", "path": str(path)}


def existing_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def generate_samples(
    *,
    base_url: str,
    dataset: Mapping[str, Any],
    task_ids: Sequence[str],
    sanitize: Callable[..., str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples_path = out_dir / "samples.jsonl"
    raw_path = out_dir / "samples.raw.jsonl"
    receipts_path = out_dir / "generation_receipts.jsonl"
    already = existing_task_ids(samples_path) if args.resume else set()
    todo = [task_id for task_id in task_ids if task_id not in already]
    started = time.time()
    generated = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    finish_reasons: dict[str, int] = {}

    with (
        samples_path.open("a", encoding="utf-8") as samples,
        raw_path.open("a", encoding="utf-8") as raw,
        receipts_path.open("a", encoding="utf-8") as receipts,
    ):
        for task_id in todo:
            task = dataset[task_id]
            payload = build_chat_payload(
                str(task["prompt"]),
                model_id=args.model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            request_id = (
                f"humaneval-screen-{args.label}-"
                f"{str(task_id).replace('/', '-')}"
            )
            request_started = time.perf_counter()
            response = http_post(
                f"{base_url}/v1/chat/completions",
                payload,
                timeout=args.request_timeout,
                headers={"x-mtplx-request-id": request_id},
            )
            elapsed = time.perf_counter() - request_started
            choice = response["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"non-text completion for {task_id}: {content!r}")
            reasoning = choice["message"].get("reasoning_content")
            if reasoning:
                raise RuntimeError(
                    f"{task_id}: server emitted reasoning with thinking off "
                    "-- the eval contract is broken, not the kernel"
                )
            clean = sanitize(content, entrypoint=str(task["entry_point"]))
            samples.write(
                json.dumps({"task_id": task_id, "solution": clean}) + "\n"
            )
            raw.write(json.dumps({"task_id": task_id, "solution": content}) + "\n")
            usage = response.get("usage", {}) or {}
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0) or 0)
            finish = str(choice.get("finish_reason"))
            finish_reasons[finish] = finish_reasons.get(finish, 0) + 1
            receipts.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "elapsed_s": elapsed,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": usage,
                        "request_id": request_id,
                        "response_id": response.get("id"),
                        "sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                + "\n"
            )
            for handle in (samples, raw, receipts):
                handle.flush()
            generated += 1
            print(
                f"[humaneval-screen:{args.label}] "
                f"{len(already) + generated}/{len(task_ids)} {task_id} "
                f"{elapsed:.2f}s finish={finish} "
                f"tokens={usage.get('completion_tokens')}",
                flush=True,
            )
    return {
        "samples": str(samples_path),
        "raw": str(raw_path),
        "generation_receipts": str(receipts_path),
        "generated": generated,
        "resumed": len(already),
        "wall_s": time.time() - started,
        "usage_total": usage_total,
        "finish_reasons": finish_reasons,
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def write_scoring_file(
    samples_path: Path, scoring_path: Path, all_task_ids: Iterable[str]
) -> dict[str, Any]:
    """Pad the samples to all 164 problems so evalplus.evaluate's assert passes."""

    rows: dict[str, str] = {}
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["task_id"])] = str(row["solution"])
    padded: list[str] = []
    with scoring_path.open("w", encoding="utf-8") as handle:
        for task_id in all_task_ids:
            solution = rows.get(task_id)
            if solution is None:
                solution = ""
                padded.append(task_id)
            handle.write(
                json.dumps({"task_id": task_id, "solution": solution}) + "\n"
            )
    return {
        "path": str(scoring_path),
        "scored": len(rows),
        "padded": len(padded),
        "padded_task_ids": padded,
    }


def score_samples(scoring_path: Path, *, parallel: int, timeout: float) -> Path:
    """Run the evalplus.evaluate CLI in the evalplus checkout's own venv."""

    result_path = Path(str(scoring_path).replace(".jsonl", "_eval_results.json"))
    if result_path.exists():
        result_path.unlink()
    command = [
        str(EVALPLUS_EVALUATE),
        "humaneval",
        "--samples",
        str(scoring_path),
        "--parallel",
        str(int(parallel)),
        "--i-just-wanna-run",
    ]
    print(f"[humaneval-screen] scoring: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(EVALPLUS_ROOT),
        timeout=timeout,
        check=False,
        # evalplus prompts on stdin when the results file already exists. The
        # unlink above is what prevents that; DEVNULL turns a surprise into an
        # immediate EOFError instead of a hang inside the guarded window.
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evalplus.evaluate exited {completed.returncode} for {scoring_path}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"evalplus.evaluate wrote no results at {result_path}")
    return result_path


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


def model_provenance(model: Path) -> dict[str, Any]:
    runtime_path = model / "mtplx_runtime.json"
    source_path = model / ".mtplx-source.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    speed = runtime.get("speed_evidence") or {}
    return {
        "path": str(model),
        "public_model_id": runtime.get("public_model_id"),
        "mtplx_version": runtime.get("mtplx_version"),
        "arch_id": runtime.get("arch_id"),
        "recommended_profile": runtime.get("recommended_profile"),
        "mtp_depth_max": runtime.get("mtp_depth_max"),
        "pack_sampler": runtime.get("sampler"),
        "artifact_fingerprint": speed.get("artifact_fingerprint"),
        "repo_id": source.get("repo_id"),
        "resolved_sha": source.get("resolved_sha"),
        "revision": source.get("revision"),
        "engine_version": source.get("engine_version"),
        "mtplx_runtime_json_sha256": hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest(),
    }


def build_receipt(
    *,
    args: argparse.Namespace,
    guard: Mapping[str, Any],
    candidate_env: Mapping[str, str],
    server_argv: Sequence[str],
    provenance: Mapping[str, Any],
    health: Mapping[str, Any],
    settings: Mapping[str, Any],
    generation: Mapping[str, Any],
    scoring_file: Mapping[str, Any],
    scores: Mapping[str, Any],
    timings: Mapping[str, float],
    server_log: Mapping[str, Any],
    warmup: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "mtplx-fable-humaneval-screen-v1",
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": "candidate" if candidate_env else "control",
        "flags": {
            "candidate_env": dict(candidate_env),
            "control_family_env": dict(CONTROL_FAMILY_ENV),
            "never_exported": dict(NEVER_EXPORT),
            "server_argv": list(server_argv),
        },
        "model": dict(provenance),
        "guard": dict(guard),
        "sampler": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": 1,
            "greedy": args.temperature <= 0,
            "prompt": "evalplus-0.3.1-openai-chat",
            "enable_thinking": False,
            "server_defaults": {
                "temperature": settings.get("temperature"),
                "top_p": settings.get("top_p"),
                "top_k": settings.get("top_k"),
                "draft_temperature": settings.get("draft_temperature"),
                "reasoning": settings.get("reasoning"),
                "reasoning_effort": settings.get("reasoning_effort"),
            },
        },
        "dataset": {
            "name": "humaneval",
            "evalplus_version": EVALPLUS_VERSION,
            "n": args.n,
            "total_problems": HUMANEVAL_TASK_COUNT,
            "selection": "first n in dataset order",
            "dataset_hash": scores.get("dataset_hash"),
        },
        "scores": {
            "tasks": scores["tasks"],
            "humaneval": scores["humaneval"],
            "humaneval_plus": scores["humaneval_plus"],
            "base_failures": scores["base_failures"],
            "plus_failures": scores["plus_failures"],
        },
        "per_problem": scores["per_problem"],
        "generation": dict(generation),
        "scoring_file": dict(scoring_file),
        "server_health": dict(health),
        "server_settings": dict(settings),
        "server_warmup": dict(warmup),
        "server_log": dict(server_log),
        "timings_s": dict(timings),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--n",
        type=int,
        choices=(20, HUMANEVAL_TASK_COUNT),
        default=HUMANEVAL_TASK_COUNT,
        help="164 is the only verdict-grade size; 20 is a smoke.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--receipt-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--python", type=Path, default=VENV_PYTHON)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--server-ready-timeout", type=float, default=1200.0)
    parser.add_argument("--warmup-timeout", type=float, default=900.0)
    parser.add_argument("--score-parallel", type=int, default=8)
    parser.add_argument("--score-timeout", type=float, default=1800.0)
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
        help="Print the outer guarded command and the server argv, then exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_env = parse_env_settings(args.env)
    server_argv = build_server_argv(
        python=args.python,
        model=args.model,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
    )

    if args.dry_run:
        print("[humaneval-screen] outer command:")
        print("  " + outer_command_line(
            label=args.label, candidate_env=args.env, n=args.n, port=args.port
        ))
        print("[humaneval-screen] server argv:")
        print("  " + " ".join(server_argv))
        print("[humaneval-screen] candidate env: " + json.dumps(candidate_env))
        return 0

    driver = load_abba_driver()
    guard = driver.acquire_guard(args.guard_mode)

    provenance = model_provenance(args.model)
    if provenance["resolved_sha"] != EXPECTED_MODEL_SHA:
        raise RuntimeError(
            f"model artifact moved: expected {EXPECTED_MODEL_SHA}, "
            f"observed {provenance['resolved_sha']}"
        )
    if not port_is_free(args.host, args.port):
        raise RuntimeError(
            f"{args.host}:{args.port} already has a listener; refusing to overlap"
        )

    available = driver.wait_for_memory()
    print(
        f"[humaneval-screen] lock attested ({guard['mode']}); "
        f"reclaimable={available / 1024**3:.2f} GiB",
        flush=True,
    )

    run_dir = args.receipt_dir.resolve() / f"{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    receipt_path = run_dir.parent / f"{args.label}.json"
    arm_claim = claim_arm_identity(
        run_dir, arm_identity(candidate_env, server_argv, args)
    )
    print(f"[humaneval-screen] arm identity {arm_claim['claimed']}", flush=True)

    get_humaneval, sanitize = load_evalplus()
    dataset = get_humaneval()
    all_task_ids = list(dataset)
    task_ids = select_task_ids(all_task_ids, args.n)

    environment = build_server_env(os.environ, candidate_env)
    base_url = f"http://{args.host}:{args.port}"
    timings: dict[str, float] = {}
    run_started = time.time()

    arm = "candidate" if candidate_env else "control"
    print(
        f"[humaneval-screen] arm={arm} env={json.dumps(candidate_env, sort_keys=True)}",
        flush=True,
    )
    print("[humaneval-screen] " + " ".join(server_argv), flush=True)

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
            health = wait_for_health(
                base_url,
                process=process,
                log_path=log_path,
                timeout=args.server_ready_timeout,
            )
            settings = http_get(f"{base_url}/v1/mtplx/settings", timeout=15.0)
            assert_server_contract(health, settings)

            # Reuse the guard's own background-warmup wait so the eval never
            # races startup GPU work. It REQUIRES health.warmup.background; if
            # this build does not publish it, say so in the receipt instead of
            # pretending we waited. A warmup that actually FAILED still raises.
            background = ((health.get("warmup") or {}).get("background"))
            if isinstance(background, dict):
                warmup_health = load_run_guarded().wait_for_background_warmup(
                    base_url,
                    timeout=args.warmup_timeout,
                    fetch=lambda url: http_get(f"{url}/health", timeout=15.0),
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
                    "[humaneval-screen] no warmup.background in /health; "
                    "not waiting",
                    flush=True,
                )

            ready = http_post(
                f"{base_url}/v1/chat/completions",
                build_chat_payload(
                    "def ready():\n    \"\"\"Return the string READY.\"\"\"\n",
                    model_id=args.model_id,
                    max_tokens=64,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
                timeout=300.0,
            )
            ready_text = ready["choices"][0]["message"].get("content")
            if not isinstance(ready_text, str) or not ready_text.strip():
                raise RuntimeError(f"READY chat returned no content: {ready!r}")
            timings["server_ready_s"] = time.time() - server_started
            print(
                f"[humaneval-screen] ready in {timings['server_ready_s']:.0f}s "
                f"({len(ready_text)} chars)",
                flush=True,
            )

            generation = generate_samples(
                base_url=base_url,
                dataset=dataset,
                task_ids=task_ids,
                sanitize=sanitize,
                out_dir=run_dir,
                args=args,
            )
            timings["generation_s"] = generation["wall_s"]
    finally:
        if process is not None:
            stop_receipt = stop_server(process)
            print(f"[humaneval-screen] server stopped: {stop_receipt}", flush=True)

    scoring_started = time.time()
    scoring_file = write_scoring_file(
        run_dir / "samples.jsonl", run_dir / "samples_scored.jsonl", all_task_ids
    )
    result_path = score_samples(
        Path(scoring_file["path"]),
        parallel=args.score_parallel,
        timeout=args.score_timeout,
    )
    eval_results = json.loads(result_path.read_text(encoding="utf-8"))
    scores = summarize_scores(eval_results, task_ids)
    scores["dataset_hash"] = eval_results.get("hash")
    timings["scoring_s"] = time.time() - scoring_started
    timings["total_s"] = time.time() - run_started

    receipt = build_receipt(
        args=args,
        guard=guard,
        candidate_env=candidate_env,
        server_argv=server_argv,
        provenance=provenance,
        health=health,
        settings=settings,
        generation=generation,
        scoring_file=scoring_file,
        scores=scores,
        timings=timings,
        server_log={
            "path": str(log_path),
            "stop": stop_receipt,
            "tail": tail_lines(log_path, 200),
        },
        warmup=warmup,
    )
    receipt["scoring"] = {"eval_results": str(result_path)}
    receipt["arm_identity"] = dict(arm_claim)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    base = scores["humaneval"]
    plus = scores["humaneval_plus"]
    print(
        f"[humaneval-screen] {args.label}: HumanEval "
        f"{base['passed']}/{scores['tasks']} = {base['pass_at_1']:.4f}; "
        f"HumanEval+ {plus['passed']}/{scores['tasks']} = "
        f"{plus['pass_at_1']:.4f}",
        flush=True,
    )
    print(f"wrote {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
