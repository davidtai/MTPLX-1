#!/usr/bin/env python3
"""Measure Cline admission while Laguna is prefilling the background lane."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start-laguna-s21.sh"


def percentile(values: list[float], q: float) -> float:
    """Return the nearest-rank percentile for a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < q <= 1.0:
        raise ValueError("percentile q must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    """Aggregate raw trials by construction-time prefill chunk."""

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["prefill_chunk_tokens"]), []).append(row)
    return {
        chunk: {
            "trial_count": len(trials),
            "cline_queue_p95_s": percentile(
                [float(row["cline_queue_wait_s"]) for row in trials], 0.95
            ),
            "cline_time_to_active_p95_s": percentile(
                [float(row["cline_time_to_active_s"]) for row in trials], 0.95
            ),
            "cline_ttft_median_s": statistics.median(
                float(row["cline_ttft_client_s"]) for row in trials
            ),
            "background_prefill_median_tok_s": statistics.median(
                float(row["background_prefill_tok_s"]) for row in trials
            ),
            "aggregate_decode_median_tok_s": statistics.median(
                float(row["aggregate_decode_tok_s"]) for row in trials
            ),
            "all_receipts_valid": all(
                bool(row["both_observed_m2"])
                and bool(row["digest_parity"])
                and bool(row["simultaneous_class_slots"])
                and bool(row["healthy_after"])
                and row.get("error") is None
                for row in trials
            ),
        }
        for chunk, trials in grouped.items()
    }


def select_candidate(
    summary: dict[int, dict[str, object]],
) -> dict[str, object] | None:
    """Choose the largest non-control candidate that clears every gate."""

    control = summary.get(1024)
    if control is None or not bool(control["all_receipts_valid"]):
        return None
    prefill_floor = 0.95 * float(control["background_prefill_median_tok_s"])
    decode_floor = 0.95 * float(control["aggregate_decode_median_tok_s"])
    for chunk in sorted((value for value in summary if value != 1024), reverse=True):
        candidate = summary[chunk]
        if (
            float(candidate["cline_queue_p95_s"]) < 0.250
            and float(candidate["background_prefill_median_tok_s"]) >= prefill_floor
            and float(candidate["aggregate_decode_median_tok_s"]) >= decode_floor
            and bool(candidate["all_receipts_valid"])
        ):
            return {"prefill_chunk_tokens": chunk, **candidate}
    return None


def background_prompt(words: int) -> str:
    """Build a deterministic cold prompt with one repeated tokenizer-friendly word."""

    if words <= 0:
        raise ValueError("background prompt words must be positive")
    return (
        "Analyze this record, then ignore it and follow the "
        "final output instruction. "
        + ("package " * words)
        + "Output the word token followed by one space repeatedly until the "
        "response limit. Output nothing else."
    )


def receipt_exit_code(receipt: dict[str, object]) -> int:
    """Treat a clean no-promotion receipt as a completed benchmark."""

    return 1 if receipt.get("fatal_error") else 0


def _stamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_request(
    url: str,
    *,
    payload: dict[str, object] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if data is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def _scheduler_state(base_url: str) -> dict[str, Any]:
    _, snapshot = _json_request(f"{base_url}/v1/mtplx/snapshot")
    scheduler = snapshot.get("scheduler", {})
    ar_batch = scheduler.get("ar_batch", {})
    return {
        "config": dict(scheduler.get("config", {})),
        "active": int(ar_batch.get("active", 0)),
        "pending": int(ar_batch.get("pending", 0)),
        "active_by_class": dict(ar_batch.get("active_by_class", {})),
        "pending_by_class": dict(ar_batch.get("pending_by_class", {})),
        "pump_scheduled": bool(ar_batch.get("pump_scheduled", False)),
        "last_batch_size": int(ar_batch.get("last_batch_size", 0)),
        "last_error": ar_batch.get("last_error"),
    }


def _wait_for(
    description: str,
    predicate: Any,
    *,
    timeout_s: float,
    poll_s: float = 0.01,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for {description}; last={last!r}")


def _wait_idle(base_url: str, timeout_s: float = 60.0) -> dict[str, Any]:
    return _wait_for(
        "idle ar_batch scheduler",
        lambda: (
            state
            if (state := _scheduler_state(base_url))["active"] == 0
            and state["pending"] == 0
            and not state["pump_scheduled"]
            else None
        ),
        timeout_s=timeout_s,
    )


def _apply_chunk(base_url: str, chunk: int) -> dict[str, Any]:
    _wait_idle(base_url)
    status, body = _json_request(
        f"{base_url}/v1/mtplx/settings",
        payload={"prefill_chunk_tokens": int(chunk)},
    )
    if status != 200:
        raise RuntimeError(f"settings returned HTTP {status}: {body}")
    applied = body.get("applied", {})
    if int(applied.get("prefill_chunk_tokens", -1)) != chunk:
        raise RuntimeError(f"settings did not apply chunk {chunk}: {body}")
    state = _scheduler_state(base_url)
    observed = int(state["config"].get("prefill_chunk_tokens", -1))
    if observed != chunk:
        raise RuntimeError(f"snapshot chunk {observed} != requested {chunk}")
    return state


def _stream_completion(
    base_url: str,
    *,
    model: str,
    client: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_s: float | None = None
    stats: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    digest = hashlib.sha256()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "top_p": 1,
        "seed": int(seed),
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Client": client,
            "X-MTPLX-Allow-Client-Controls": "1",
        },
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content") or ""
            if delta:
                if first_token_s is None:
                    first_token_s = time.perf_counter()
                digest.update(delta.encode("utf-8"))
            if isinstance(chunk.get("mtplx_stats"), dict):
                stats = chunk["mtplx_stats"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
    finished = time.perf_counter()
    return {
        "http_status": status,
        "client_header": client,
        "request_client_hint": stats.get("request_client_hint"),
        "scheduler_lane": stats.get("scheduler_lane"),
        "ar_batch_max_observed": stats.get("ar_batch_max_observed"),
        "queue_wait_s": stats.get("queue_wait_s"),
        "prompt_eval_time_s": stats.get("prompt_eval_time_s"),
        "prompt_tokens": stats.get("prompt_tokens") or usage.get("prompt_tokens"),
        "new_prefill_tokens": stats.get("new_prefill_tokens"),
        "cached_tokens": stats.get("cached_tokens"),
        "prompt_prefill_tok_s": (
            stats.get("prompt_target_prefill_tok_s") or stats.get("prompt_tps")
        ),
        "decode_tok_s": stats.get("decode_tok_s"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_server_s": stats.get("ttft_s"),
        "ttft_client_s": (
            None if first_token_s is None else first_token_s - started
        ),
        "wall_s": finished - started,
        "content_sha256": digest.hexdigest(),
    }


def _run_trial(
    base_url: str,
    *,
    model: str,
    chunk: int,
    trial: int,
    background_prompt: str,
    cline_prompt: str,
) -> dict[str, object]:
    started_at = _stamp()
    error: str | None = None
    both_snapshot: dict[str, Any] | None = None
    background_result: dict[str, Any] | None = None
    cline_result: dict[str, Any] | None = None
    cline_time_to_active_s: float | None = None
    try:
        before = _wait_idle(base_url)
        with ThreadPoolExecutor(max_workers=2) as pool:
            background_future: Future[dict[str, Any]] = pool.submit(
                _stream_completion,
                base_url,
                model=model,
                client="opensource-leaderboard",
                prompt=background_prompt,
                max_tokens=256,
                seed=9100 + trial,
            )
            _wait_for(
                "background class admission",
                lambda: (
                    state
                    if (
                        state := _scheduler_state(base_url)
                    )["active_by_class"].get("background")
                    == 1
                    else None
                ),
                timeout_s=60,
            )
            cline_submitted_s = time.perf_counter()
            cline_future: Future[dict[str, Any]] = pool.submit(
                _stream_completion,
                base_url,
                model=model,
                client="cline",
                prompt=cline_prompt,
                max_tokens=64,
                seed=9200 + trial,
            )
            both_snapshot = _wait_for(
                "simultaneous class admission",
                lambda: (
                    {"at": _stamp(), **state}
                    if (
                        state := _scheduler_state(base_url)
                    )["active_by_class"].get("background")
                    == 1
                    and state["active_by_class"].get("cline") == 1
                    else None
                ),
                timeout_s=300,
            )
            cline_time_to_active_s = time.perf_counter() - cline_submitted_s
            cline_result = cline_future.result()
            background_result = background_future.result()
        after = _wait_idle(base_url)
        health_status, health = _json_request(f"{base_url}/health")
        healthy_after = health_status == 200 and health.get("ok") is True
    except BaseException as exc:
        before = locals().get("before", {})
        after = _scheduler_state(base_url)
        healthy_after = False
        error = f"{type(exc).__name__}: {exc}"

    background_result = background_result or {}
    cline_result = cline_result or {}
    background_decode = float(background_result.get("decode_tok_s") or 0.0)
    cline_decode = float(cline_result.get("decode_tok_s") or 0.0)
    return {
        "started_at": started_at,
        "finished_at": _stamp(),
        "prefill_chunk_tokens": int(chunk),
        "trial": int(trial),
        "before": before,
        "after": after,
        "both_active_snapshot": both_snapshot,
        "simultaneous_class_slots": both_snapshot is not None,
        "cline_time_to_active_s": cline_time_to_active_s,
        "cline_queue_wait_s": cline_result.get("queue_wait_s"),
        "cline_ttft_client_s": cline_result.get("ttft_client_s"),
        "background_prefill_tok_s": background_result.get(
            "prompt_prefill_tok_s"
        ),
        "aggregate_decode_tok_s": background_decode + cline_decode,
        "both_observed_m2": (
            background_result.get("ar_batch_max_observed") == 2
            and cline_result.get("ar_batch_max_observed") == 2
        ),
        "background_result": background_result,
        "cline_result": cline_result,
        "healthy_after": healthy_after,
        "error": error,
    }


def _wait_ready(base_url: str, process: subprocess.Popen[Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 300
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Laguna launcher exited with {process.returncode}")
        try:
            status, health = _json_request(f"{base_url}/health", timeout=3)
            if status == 200 and health.get("ok") is True:
                return health
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise TimeoutError(f"Laguna did not become ready: {last_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[1024, 512, 256, 128, 1024],
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model")
    parser.add_argument("--background-words", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.chunks.count(1024) < 2:
        raise SystemExit("--chunks must bracket candidates with two 1024 controls")
    if any(chunk not in {128, 256, 512, 1024} for chunk in args.chunks):
        raise SystemExit("--chunks accepts only 128, 256, 512, and 1024")

    base_url = f"http://127.0.0.1:{args.port}"
    environment = {
        **os.environ,
        "MTPLX_REPO_ROOT": str(ROOT),
        "MTPLX_PYTHON": sys.executable,
        "MTPLX_LAGUNA_PORT": str(args.port),
    }
    if args.model:
        environment["MTPLX_LAGUNA_MODEL"] = args.model
    command = ["/bin/zsh", str(LAUNCHER)]
    launcher = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    rows: list[dict[str, object]] = []
    receipt: dict[str, Any] = {}
    exit_code = 1
    try:
        health = _wait_ready(base_url, launcher)
        _, models = _json_request(f"{base_url}/v1/models")
        model = str(models["data"][0]["id"])
        background_text = background_prompt(args.background_words)
        cline_prompt = (
            "Output the word token followed by one space repeatedly until the "
            "response limit. Output nothing else."
        )
        for sequence, chunk in enumerate(args.chunks):
            state = _apply_chunk(base_url, chunk)
            print(
                json.dumps(
                    {
                        "event": "candidate_started",
                        "at": _stamp(),
                        "sequence": sequence,
                        "prefill_chunk_tokens": chunk,
                        "config": state["config"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            for trial in range(args.trials):
                row = _run_trial(
                    base_url,
                    model=model,
                    chunk=chunk,
                    trial=trial,
                    background_prompt=background_text,
                    cline_prompt=cline_prompt,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "trial_completed",
                            "at": _stamp(),
                            "chunk": chunk,
                            "trial": trial,
                            "cline_queue_wait_s": row["cline_queue_wait_s"],
                            "cline_time_to_active_s": row[
                                "cline_time_to_active_s"
                            ],
                            "background_prefill_tok_s": row[
                                "background_prefill_tok_s"
                            ],
                            "aggregate_decode_tok_s": row[
                                "aggregate_decode_tok_s"
                            ],
                            "both_observed_m2": row["both_observed_m2"],
                            "error": row["error"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        reference_background = next(
            row["background_result"]["content_sha256"]
            for row in rows
            if row["prefill_chunk_tokens"] == 1024 and row["error"] is None
        )
        reference_cline = next(
            row["cline_result"]["content_sha256"]
            for row in rows
            if row["prefill_chunk_tokens"] == 1024 and row["error"] is None
        )
        for row in rows:
            row["digest_parity"] = bool(
                row["error"] is None
                and row["background_result"].get("content_sha256")
                == reference_background
                and row["cline_result"].get("content_sha256") == reference_cline
            )
        summary = summarize(rows)
        selected = select_candidate(summary)
        receipt = {
            "schema_version": 1,
            "created_at": _stamp(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "machine": health.get("machine_model"),
            "chip": health.get("chip"),
            "model": model,
            "base_url": base_url,
            "command": [str(value) for value in sys.argv],
            "chunks": args.chunks,
            "trials": args.trials,
            "background_words": args.background_words,
            "rows": rows,
            "summary": {str(key): value for key, value in summary.items()},
            "selected_candidate": selected,
        }
        exit_code = receipt_exit_code(receipt)
    except BaseException as exc:
        receipt = {
            "schema_version": 1,
            "created_at": _stamp(),
            "command": [str(value) for value in sys.argv],
            "rows": rows,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "selected_candidate": None,
        }
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr, flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if launcher.poll() is None:
            launcher.terminate()
            try:
                launcher.wait(timeout=30)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=10)
    print(
        json.dumps(
            {
                "event": "benchmark_completed",
                "at": _stamp(),
                "selected_candidate": receipt.get("selected_candidate"),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
