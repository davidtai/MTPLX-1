#!/usr/bin/env python3
"""One measured Qwen3.8 Flash-Next decode arm for the fable ABBA harness.

This is a branch-owned adaptation of the reviewed PR391 driver
(``/private/tmp/pr391_fixed_d3_abba.py``, SHA-256
``0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0``).  It
keeps every protection that made that driver trustworthy and drops only the
``/private/tmp`` SHA pin, because this file *is* the reviewed driver for this
branch:

* guard consumption (attestation FD as a direct child, or the DeepSeek-style
  verified window receipt as a guarded grandchild -- see ``--guard-mode``),
* the reclaimable-memory wait before any model load,
* the thermal gate, ON by default at 40 C and adjustable only through
  ``--thermal-gate-max-c`` (there is no flag that silently disables it),
* the turbo profile environment plus family overrides plus ``--candidate-env``,
  with the same fail-closed "effective construction environment drifted" check,
* the production benchmark cell (16,384-token coding prompt, 1,024 output
  tokens, temperature 1 / top-p 0.95 / top-k 20, reasoning ``xhigh``), which is
  byte-identical to ``build_benchmark_cells``' ``coding-16k-1k-xhigh-t1`` cell.
  ``--prompt-tokens`` resizes that prompt (the label, the receipt path and
  every other cell parameter are unchanged); the default reproduces the
  pinned 16,384-token prompt byte for byte,
* ``reset_run_caches`` between measured runs.

D3 softfloat64 route
--------------------
``scripts/pr391_metal_choice_benchmark_launcher.transform_metal_choice_driver``
injects three blocks into the reviewed driver's *text*: a prebind block after
model load, an install block immediately before the timer, and an attach block
right after ``row["pre_run_reset"] = reset_receipt``.  This driver **ports the
injected code directly** rather than re-running the text transform, for two
reasons that are both about provable identity:

1. every non-trivial object in the injected code -- ``prebind_softfloat64_
   choice_kernel``, ``prewarm_softfloat64_verifier_decision``,
   ``PR391DirectSoftFloat64D3Route``, ``validate_metal_choice_receipt``,
   ``build_exact_output_parity_receipt``, ``build_hit_miss_receipt`` -- is
   *imported from that same module*, so the route, its receipts and its
   fail-closed validation are the same code, not a copy;
2. the text transform also replaces the thermal gate with
   ``{"disabled": True, "reason": "user_requested"}``.  Re-running it here
   would silently disable the gate this driver is required to keep.

The route is opt-in through ``--d3-softfloat64-route``.  It is **off by
default** because the retained 67.818 tok/s "paired routed GLU" control arm
(receipt ``rebench3-1788287001-paired-routed-glu-candidate-seeds-16k-1k-seeds-
16k-1k.json``) did not use it: that receipt has ``candidate_files == {}`` and
its row carries no ``metal_softfloat64_choice_route`` key.  Turning the route
on costs roughly 1.5-2 tok/s, so defaulting it on would make the control fail
to reproduce its own retained number.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import stat
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
# The qwen38 generation fixtures live only in the qwen4-queue-first-draft
# worktree; they are not committed on this branch.  Pinned by content hash so
# a moved or edited fixture fails loudly instead of silently changing the cell.
FIXTURES = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/"
    "qwen4-queue-first-draft/mtplx/benchmarks/prompts"
)
EXPECTED_CONTEXT_SHA256 = (
    "c8ae2b1790c0300aa7c1421b55e7cd5d43c93461f7fba5d3a732fd34e156b4c4"
)
LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
OUT_DIR = ROOT / ".benchmark-artifacts/fable"
EXPECTED_MODEL = "29ba90f82124961d0d902a9ea9bbb1034972af2f"
EXPECTED_PROMPT = "b9e9acf190a37eb12bad2171dea6bbfa4e13a8f1053ce4888e5c3cb3fadfdd20"
PRODUCTION_CELL_LABEL = "coding-16k-1k-xhigh-t1"
# The measured cell's prompt length.  16,384 is the production cell and the
# only value whose prompt bytes are pinned by ``EXPECTED_PROMPT``; every
# other value is built by ``build_exact_coding_prompt_ids`` from the same
# SHA-pinned fixture pair, which is also how ``build_benchmark_cells``
# builds its 64K/128K cells -- so ``--prompt-tokens 65536`` is the same
# prompt as the matrix's ``coding-64k-1k-xhigh-t1``.
DEFAULT_PROMPT_TOKENS = 16_384
PROMPT_TOKEN_CHOICES = (
    1_024,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
)
MIN_AVAILABLE_BYTES = 90 * 1024**3
MINIMUM_RESIDENT_BYTES = 89_480_048_859
MEMORY_LIMIT_BYTES = 96 * 1024**3
WIRED_LIMIT_BYTES = 90 * 1024**3
NGRAM_TABLE_NAME = "ngram-table.safetensors"
PREWARM_CHUNK_BYTES = 64 * 1024**2
DEFAULT_THERMAL_MAX_C = 40.0


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


def consume_guard_attestation() -> dict[str, Any]:
    """Consume the one-shot guard pipe as the guard's direct child."""

    raw_fd = os.environ.pop("MTPLX_GUARD_ATTEST_FD", None)
    nonce = os.environ.pop("MTPLX_GUARD_ATTEST_NONCE", None)
    if raw_fd is None or nonce is None:
        raise RuntimeError("canonical GPU guard attestation is required")
    descriptor = int(raw_fd)
    payload = bytearray()
    try:
        while len(payload) <= 16 * 1024:
            chunk = os.read(descriptor, 16 * 1024 + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    receipt = json.loads(payload)
    lock_path = LOCK.resolve(strict=True)
    lock_status = lock_path.lstat()
    expected = {
        "schema_version": 1,
        "nonce": nonce,
        "child_pid": os.getpid(),
        "guard_pid": os.getppid(),
        "lock_path": str(lock_path),
        "lock_device": lock_status.st_dev,
        "lock_inode": lock_status.st_ino,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"guard attestation mismatch: {key}")
    if (
        not stat.S_ISREG(lock_status.st_mode)
        or lock_status.st_nlink != 1
        or stat.S_IMODE(lock_status.st_mode) != 0o600
        or lock_status.st_uid != os.getuid()
    ):
        raise RuntimeError("canonical GPU lock identity is unsafe")
    with lock_path.open("r+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            raise RuntimeError("guard claims the lane but the GPU lock is free")
    return receipt


def consume_guard_window() -> dict[str, Any]:
    """Verify the inherited window receipt as a guarded grandchild.

    ``scripts/deepseek_v4_guard_window.py`` is the repository's working pattern
    for this: the guard's direct child consumes the one-shot pipe once and
    publishes a private, read-only receipt; every grandchild re-verifies that
    receipt against the still-live process ancestry and the still-held lock.
    """

    sys.path.insert(0, str(ROOT))
    from scripts.deepseek_v4_guard_window import load_verified_guard_window

    document = load_verified_guard_window()
    attestation = document["attestation"]
    return {
        "mode": "verified_window_receipt",
        "lock_path": attestation["lock_path"],
        "guard_pid": attestation["guard_pid"],
        "child_pid": attestation["child_pid"],
        "window_id": document["window_id"],
        "receipt_sha256": document["receipt_sha256"],
        "consumer_pid": document["consumer_verification"]["consumer_pid"],
    }


def acquire_guard(mode: str) -> dict[str, Any]:
    """Resolve the guard evidence for this process."""

    if mode == "auto":
        if os.environ.get("MTPLX_DSV4_GUARD_WINDOW_PATH"):
            mode = "window"
        elif os.environ.get("MTPLX_GUARD_ATTEST_FD"):
            mode = "attestation"
        else:
            raise RuntimeError(
                "no GPU guard evidence: run this under bench/laguna/"
                "run_guarded.py (directly, or through scripts/fable/"
                "abba_window.py)"
            )
    if mode == "window":
        return consume_guard_window()
    receipt = consume_guard_attestation()
    return {
        "mode": "attestation_fd",
        "lock_path": receipt["lock_path"],
        "guard_pid": receipt["guard_pid"],
        "child_pid": receipt["child_pid"],
    }


# --------------------------------------------------------------------------
# Memory / thermal gates
# --------------------------------------------------------------------------


def available_memory_bytes() -> int:
    output = subprocess.check_output(["vm_stat"], text=True)
    pages: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        digits = "".join(character for character in raw if character.isdigit())
        if digits:
            pages[key] = int(digits)
    return 16_384 * sum(
        pages.get(key, 0)
        for key in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        )
    )


def wait_for_memory() -> int:
    deadline = time.monotonic() + 300
    while True:
        available = available_memory_bytes()
        if available >= MIN_AVAILABLE_BYTES:
            return available
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"only {available / 1024**3:.2f} GiB reclaimable; refusing model load"
            )
        time.sleep(2)


def read_machine_temperature() -> dict[str, float]:
    process = subprocess.Popen(
        ["/opt/homebrew/bin/macmon", "pipe", "-s", "1", "-i", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("macmon did not expose stdout")
        line = process.stdout.readline()
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    payload = json.loads(line)
    temperature = payload.get("temp") or {}
    cpu_c = float(temperature["cpu_temp_avg"])
    gpu_c = float(temperature["gpu_temp_avg"])
    return {"cpu_c": cpu_c, "gpu_c": gpu_c, "max_c": max(cpu_c, gpu_c)}


def wait_for_temperature(
    max_celsius: float = DEFAULT_THERMAL_MAX_C,
) -> dict[str, Any]:
    """Block outside the timed region until macOS reports a cool CPU and GPU."""

    started = time.monotonic()
    deadline = started + 3_600
    samples = 0
    initial_celsius = None
    while True:
        reading = read_machine_temperature()
        samples += 1
        celsius = float(reading["max_c"])
        sensor = "macmon:max(cpu_temp_avg,gpu_temp_avg)"
        if initial_celsius is None:
            initial_celsius = celsius
        print(
            f"[fable-abba] thermal gate {sensor}={celsius:.1f}C "
            f"target<={max_celsius:.1f}C",
            flush=True,
        )
        if celsius <= max_celsius:
            return {
                "threshold_c": max_celsius,
                "initial_c": initial_celsius,
                "ready_c": celsius,
                "ready_cpu_c": float(reading["cpu_c"]),
                "ready_gpu_c": float(reading["gpu_c"]),
                "sensor": sensor,
                "wait_s": time.monotonic() - started,
                "samples": samples,
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"thermal gate timed out at {celsius:.1f}C on {sensor}"
            )
        time.sleep(10)


def thermal_gate_max_c(raw: str) -> float:
    """Parse ``--thermal-gate-max-c``; there is no value that disables it."""

    value = float(raw)
    if not 20.0 <= value <= 90.0:
        raise argparse.ArgumentTypeError(
            "thermal gate threshold must be within [20, 90] C; the gate "
            "cannot be disabled"
        )
    return value


# --------------------------------------------------------------------------
# Source / input verification
# --------------------------------------------------------------------------


def verify_inputs(
    source_path: Path,
    expected_source: str | None,
    expected_files: list[str],
    *,
    allow_dirty_source: bool,
    model_path: Path,
) -> dict[str, Any]:
    """Pin the benchmark source and the model revision before any load."""

    source = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_source is not None and source != expected_source:
        raise RuntimeError(
            f"source commit drifted: expected {expected_source}, got {source}"
        )
    file_hashes: dict[str, str] = {}
    for spec in expected_files:
        try:
            relative, expected = spec.split("=", 1)
        except ValueError as exc:
            raise RuntimeError(f"bad --expected-file value: {spec}") from exc
        actual = hashlib.sha256((source_path / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"candidate source drifted: {relative}")
        file_hashes[relative] = actual
    dirty = subprocess.check_output(
        ["git", "-C", str(source_path), "status", "--porcelain"], text=True
    )
    if dirty and not expected_files and not allow_dirty_source:
        raise RuntimeError(
            f"clean benchmark source is dirty: {source_path}\n{dirty}"
            "\n(pass --expected-file for the changed files, or "
            "--allow-dirty-source to record the drift instead)"
        )
    model_source = json.loads((model_path / ".mtplx-source.json").read_text())
    if model_source["resolved_sha"] != EXPECTED_MODEL:
        raise RuntimeError("model revision drifted")
    context_sha = hashlib.sha256(
        (FIXTURES / "qwen38_generation_context.py").read_bytes()
    ).hexdigest()
    if context_sha != EXPECTED_CONTEXT_SHA256:
        raise RuntimeError(
            f"benchmark fixture drifted: {FIXTURES / 'qwen38_generation_context.py'}"
        )
    return {
        "observed_commit": source,
        "expected_commit": expected_source,
        "dirty": bool(dirty),
        "dirty_entries": sorted(
            line[3:] for line in dirty.splitlines() if len(line) > 3
        ),
        "candidate_files": file_hashes,
        "fixture_context_sha256": context_sha,
    }


def assert_mtplx_tree(source_path: Path) -> str:
    """Fail loudly when the editable install resolves a different worktree."""

    import mtplx

    resolved = str(Path(mtplx.__file__).resolve())
    expected_prefix = str(source_path.resolve()) + os.sep
    if not resolved.startswith(expected_prefix):
        raise RuntimeError(
            "mtplx resolved outside the benchmark source tree: "
            f"{resolved} is not under {expected_prefix}. "
            "Export PYTHONPATH=<worktree> on the wrapper invocation."
        )
    return resolved


# --------------------------------------------------------------------------
# Page-cache prewarm
# --------------------------------------------------------------------------


def prewarm_ngram_table(path: Path) -> dict[str, Any]:
    """Read the n-gram table sequentially once so decode sees a warm cache."""

    started = time.perf_counter()
    total = 0
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        while True:
            chunk = os.read(descriptor, PREWARM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
    finally:
        os.close(descriptor)
    elapsed = time.perf_counter() - started
    size = path.stat().st_size
    if total != size:
        raise RuntimeError(
            f"n-gram prewarm read {total} of {size} bytes from {path}"
        )
    return {
        "path": str(path),
        "bytes": total,
        "seconds": elapsed,
        "chunk_bytes": PREWARM_CHUNK_BYTES,
        "gib_per_s": (total / 1024**3) / elapsed if elapsed > 0 else None,
    }


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def stats_receipt(
    output: Any, arm: str, sequence: int, seed: int, wall_s: float
) -> dict[str, Any]:
    stats = output.stats
    prompt_state_total_time_s = float(
        getattr(stats, "prompt_state_total_time_s", 0.0)
    )
    pre_first_token_setup_s = float(
        getattr(stats, "pre_first_token_setup_s", 0.0)
    )
    first_primary_sample_time_s = float(
        getattr(stats, "first_primary_sample_time_s", 0.0)
    )
    return {
        "sequence": sequence,
        "seed": seed,
        "arm": arm,
        "wall_s": wall_s,
        "generated_tokens": int(stats.generated_tokens),
        "prompt_eval_time_s": float(getattr(stats, "prompt_eval_time_s", 0.0)),
        "prefill_tok_s": float(getattr(stats, "prompt_tps", 0.0)),
        "ttft_s": (
            prompt_state_total_time_s
            + pre_first_token_setup_s
            + first_primary_sample_time_s
        ),
        "prompt_state_total_time_s": prompt_state_total_time_s,
        "pre_first_token_setup_s": pre_first_token_setup_s,
        "first_primary_sample_time_s": first_primary_sample_time_s,
        "decode_elapsed_s": float(stats.decode_elapsed_s),
        "decode_tok_s": float(stats.tok_s),
        "accepted_drafts": int(stats.accepted_drafts),
        "drafted_tokens": int(stats.drafted_tokens),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "bonus_tokens": int(stats.bonus_tokens),
        "correction_tokens": int(stats.correction_tokens),
        "verify_calls": int(stats.verify_calls),
        "mtp_forward_calls": int(stats.mtp_forward_calls),
        "draft_time_s": float(stats.draft_time_s),
        "verify_time_s": float(stats.verify_time_s),
        "verify_forward_time_s": float(stats.verify_forward_time_s),
        "verify_target_distribution_time_s": float(
            getattr(stats, "verify_target_distribution_time_s", 0.0)
        ),
        "repair_time_s": float(stats.repair_time_s),
        "accept_time_s": float(stats.accept_time_s),
        "snapshot_time_s": float(stats.snapshot_time_s),
        "rollback_time_s": float(stats.rollback_time_s),
        "commit_time_s": float(stats.commit_time_s),
        "capture_commit_time_s": float(stats.capture_commit_time_s),
        "verify_eval_time_s": float(stats.verify_eval_time_s),
        "verify_eval_unattributed_time_s": float(
            stats.verify_eval_unattributed_time_s
        ),
        "response_token_sha256": hashlib.sha256(
            ",".join(str(int(token)) for token in output.tokens).encode()
        ).hexdigest(),
        "compiled_verify": dict((stats.graphbank or {}).get("compiled_verify") or {}),
        "draft_core": dict(stats.draft_core or {}),
        "online_correction_cache": dict(stats.online_correction_cache or {}),
        "context_copy": {
            "active": bool(stats.context_copy_active),
            "probes": int(stats.context_copy_probes),
            "rounds": int(stats.context_copy_rounds),
            "drafted_tokens": int(stats.context_copy_drafted_tokens),
            "accepted_blocks": int(stats.context_copy_accepted_blocks),
            "accepted_tokens": int(stats.context_copy_accepted_tokens),
            "suspensions": int(stats.context_copy_suspensions),
            "disabled_reason": stats.context_copy_disabled_reason,
        },
    }


#: Raw ``--env`` passthrough settings, recorded into every receipt.
_EXTRA_ENVIRONMENT: dict[str, str] = {}


def _ple_prefill_lookahead_armed() -> bool:
    """Whether the candidate lane was actually armed in THIS process.

    Recorded next to its counters so a receipt can never again show an
    inert lane without also showing that it was asked to run: on
    2026-09-01 the flag arrived through --candidate-extra-env, which the
    receipt's `candidate_environment` does not carry, and the arm read as
    a plain 2 s regression.
    """

    try:
        from mtplx.ple_prefill_lookahead import ENV_FLAG

        return (os.environ.get(ENV_FLAG) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _ple_prefill_lookahead_counters() -> dict[str, int]:
    """Engagement counters for the PLE prefill lookahead lane (may be empty)."""

    try:
        from mtplx.ple_prefill_lookahead import snapshot_counters

        return snapshot_counters()
    except Exception:
        return {}


def _ple_prefill_lookahead_scope_status() -> dict[str, Any]:
    """Whether the LAST prefill actually ran the lane, and if not, why.

    ``prefill_lookahead_armed`` above answers a different question -- was the
    environment flag set in this process -- and must keep answering it, since
    that is the 2026-09-01 blind spot.  This answers the per-request one: a
    prefill of one chunk has nothing to look ahead from, so the scope skips
    the worker and records ``{"armed": False, "reason": "single_span"}``
    rather than reporting a non-engagement.
    """

    try:
        from mtplx.ple_prefill_lookahead import last_scope_status

        return last_scope_status()
    except Exception:
        return {}


def _ple_first_gather_early_armed() -> bool:
    """Whether MTPLX_FABLE_PLE_FIRST_GATHER_EARLY was set in THIS process.

    Same reason as ``prefill_lookahead_armed``: the flag rides
    ``--candidate-extra-env``, which ``candidate_environment`` does not carry,
    so without this a receipt could show an inert lane and no sign that the
    lane was ever asked to run.
    """

    try:
        from mtplx.ple_prefill_lookahead import EARLY_ENV_FLAG

        return (os.environ.get(EARLY_ENV_FLAG) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _ple_first_gather_early_status() -> dict[str, Any]:
    """What the LAST request's first-chunk gather did.

    ``started_at_ms_before_layer2`` is the head start the lane actually
    bought: milliseconds between the worker submit at request arrival and the
    moment the owner thread first needed the rows.  ``path`` is ``vectorized``
    when mincore found the rows' pages already in core (the fancy index alone,
    0.44 ms) and ``pread`` when it did not (the shipped threaded warm pass).
    ``outcome`` names the consumer -- ``adopted_hit`` through the lookahead's
    slot 0, ``hit`` on a single-chunk prefill where the lookahead is inert,
    and a ``miss_*``/``never_needed`` whenever the prediction did not survive
    contact with the prefill loop.
    """

    try:
        from mtplx.ple_prefill_lookahead import last_early_status

        return last_early_status()
    except Exception:
        return {}


def _ple_candidate_prefetch_armed() -> bool:
    """Whether MTPLX_FABLE_PLE_CANDIDATE_PREFETCH was set in THIS process.

    Same reason as the two lanes above: the flag rides
    ``--candidate-extra-env``, which ``candidate_environment`` does not carry,
    so without this a receipt could show an inert lane with no sign that the
    lane was ever asked to run.
    """

    try:
        from mtplx.ple_candidate_prefetch import ENV_FLAG

        return (os.environ.get(ENV_FLAG) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _ple_candidate_prefetch_receipt() -> dict[str, float]:
    """K-P1 engagement: what the candidate row buffer actually served.

    ``hits`` / ``misses`` count WINDOWS (one verify each), not rows:
    ``resolve`` is all-or-nothing, so a window with one uncovered row takes
    the shipped gather whole.  ``rows_served`` / ``rows_missing`` are the row
    detail behind that.  ``worker_wait_ms`` is the owner thread's time blocked
    joining the workers -- if that is not far below the gather it replaced,
    the lane bought nothing and the receipt says so.
    """

    try:
        from mtplx.ple_candidate_prefetch import last_receipt

        return last_receipt()
    except Exception:
        return {}


def prefill_chunks_receipt() -> list[dict[str, float]]:
    """Per-chunk prefill wall and PLE-gather seconds, on BOTH arms.

    Recorded by the chunked prefill loop itself, so a control arm carries it
    too.  This is what shows whether a run-to-run prefill swing lives in GPU
    work or in the host-late PLE gathers -- the 2026-09-01 window had a
    control spread of 13.76-15.61 s with no code difference at all.
    """

    try:
        from mtplx.generation import prefill_chunk_records

        return prefill_chunk_records()
    except Exception:
        return []


def first_chunk_cold_s(row: Mapping[str, Any] | None) -> float | None:
    """Chunk-1 wall from an unmeasured graph-warm-up row, or ``None``.

    The 2026-09-01 finding: on a fresh process the FIRST prefill chunk is
    bimodal (~1.9 s or ~4.4 s) while chunks 2-8 hold +-0.02 s, and the mode
    is perfectly concordant with the throughput of the driver's own 29.8 GiB
    ``--prewarm-ngram-table`` read (12.7 GiB/s vs ~6.5 GiB/s, 12/12 arms in
    the w22 window).  That is memory-residency state, not the candidate.
    ``--warm-graph`` moves it out of the measured run; this keeps the raw
    cold number so TTFT-after-restart stays trackable.
    """

    if not row:
        return None
    chunks = row.get("prefill_chunks") or []
    if not chunks:
        return None
    try:
        return float(chunks[0]["wall_s"])
    except (KeyError, TypeError, ValueError):
        return None


def ple_hot_rows_receipt(runtime: Any) -> dict[str, Any]:
    """PLE hot-row cache counters, as injected by pr391_current_profile_launcher."""

    try:
        text = getattr(runtime.model, "language_model", runtime.model)
        inner = text.model
        layer = inner.layers[int(inner._ple_stage_idx)].ple
        sidecar = layer.ple_embedding.ngram_embedding._sidecar
        total = int(sidecar.hot_hits + sidecar.hot_misses)
        return {
            "available": True,
            "hits": int(sidecar.hot_hits),
            "misses": int(sidecar.hot_misses),
            "hit_rate": float(sidecar.hot_hits / total) if total else None,
            "capacity_rows": int(sidecar._hot_cap_rows),
            "row_bytes": int(sidecar._hot_row_bytes),
            "capacity_bytes": int(sidecar._hot_cap_rows * sidecar._hot_row_bytes),
            "prefetch_batches": int(sidecar.prefetch_batches),
            # Worker-thread warm batches (MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD).
            # Counted apart from prefetch_batches so the candidate arm's
            # engagement is readable without reinterpreting the control's.
            "lookahead_batches": int(getattr(sidecar, "lookahead_batches", 0)),
            "prefill_lookahead": _ple_prefill_lookahead_counters(),
            "prefill_lookahead_armed": _ple_prefill_lookahead_armed(),
            "prefill_lookahead_scope": _ple_prefill_lookahead_scope_status(),
            "ple_first_gather_early": _ple_first_gather_early_status(),
            "ple_first_gather_early_armed": _ple_first_gather_early_armed(),
            "ple_candidate_prefetch": _ple_candidate_prefetch_receipt(),
            "ple_candidate_prefetch_armed": _ple_candidate_prefetch_armed(),
            # Which gather path each big row read actually took.  The pread
            # warm pass costs ~165 ms per 32,768 rows against 0.44 ms for the
            # fancy index behind it, so an arm that claims the vectorised lane
            # and shows pread_gathers is claiming a win it did not take.
            "vectorized_gathers": int(getattr(sidecar, "vectorized_gathers", 0)),
            "pread_gathers": int(getattr(sidecar, "pread_gathers", 0)),
            "madvise": getattr(sidecar, "madvise_applied", None),
            "prewarm_at_load": getattr(sidecar, "prewarm_at_load", None),
        }
    except Exception as error:  # diagnostic field, never the measurement
        print(
            f"[fable-abba] WARNING: PLE hot-row receipt unavailable: {error!r}",
            flush=True,
        )
        return {"available": False, "reason": repr(error)}


def per_cycle_receipt(stats: Any) -> dict[str, Any]:
    """Per-cycle arrays, when the engine kept per-round events.

    ``stats.events`` is populated only with ``MTPLX_DROP_EVENTS=0`` (the turbo
    profile sets it to 1), which ``--retain-events`` arranges.  Without it the
    arrays are absent rather than fabricated.
    """

    events = list(getattr(stats, "events", ()) or ())
    if not events:
        return {
            "available": False,
            "reason": "stats.events is empty (MTPLX_DROP_EVENTS=1)",
            "source": "stats.events",
        }
    steps: list[int | None] = []
    accepted: list[int | None] = []
    attributed_s: list[float | None] = []
    timing_totals: dict[str, float] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        step = event.get("step")
        steps.append(int(step) if isinstance(step, int) else None)
        accepted_value = event.get("accepted")
        accepted.append(
            int(accepted_value) if isinstance(accepted_value, int) else None
        )
        timing = event.get("timing_s")
        if isinstance(timing, dict) and timing:
            total = 0.0
            for key, value in timing.items():
                seconds = float(value)
                total += seconds
                timing_totals[key] = timing_totals.get(key, 0.0) + seconds
            attributed_s.append(total)
        else:
            attributed_s.append(None)
    return {
        "available": True,
        "source": "stats.events",
        "cycles": len(steps),
        "step": steps,
        "accepted": accepted,
        "attributed_s": attributed_s,
        "attributed_ms": [
            None if value is None else value * 1000.0 for value in attributed_s
        ],
        "timing_totals_s": dict(sorted(timing_totals.items())),
    }


def reference_token_parity(row: dict[str, Any]) -> dict[str, Any]:
    """Compare this row against the retained PR391 reference for its seed."""

    from scripts.pr391_metal_choice_benchmark_launcher import (
        COUNTER_FIELDS,
        REFERENCE_ROWS,
    )

    seed = int(row["seed"])
    reference = REFERENCE_ROWS.get(seed)
    if reference is None:
        return {"status": "no_reference", "seed": seed, "match": None}
    if len(row["accepted_by_depth"]) != len(
        reference["accepted_by_depth"]
    ) or len(row["drafted_by_depth"]) != len(reference["drafted_by_depth"]):
        return {
            "status": "depth_shape_mismatch",
            "seed": seed,
            "match": False,
            "observed_depths": len(row["accepted_by_depth"]),
            "expected_depths": len(reference["accepted_by_depth"]),
        }
    counter_deltas = {
        field: int(row[field]) - int(reference[field]) for field in COUNTER_FIELDS
    }
    accepted_deltas = [
        int(observed) - int(expected)
        for observed, expected in zip(
            row["accepted_by_depth"], reference["accepted_by_depth"], strict=True
        )
    ]
    drafted_deltas = [
        int(observed) - int(expected)
        for observed, expected in zip(
            row["drafted_by_depth"], reference["drafted_by_depth"], strict=True
        )
    ]
    digest_match = str(row["response_token_sha256"]) == str(
        reference["response_token_sha256"]
    )
    match = (
        digest_match
        and all(delta == 0 for delta in counter_deltas.values())
        and all(delta == 0 for delta in (*accepted_deltas, *drafted_deltas))
    )
    return {
        "status": "match" if match else "drift",
        "seed": seed,
        "match": match,
        "digest_match": digest_match,
        "expected_response_token_sha256": str(reference["response_token_sha256"]),
        "observed_response_token_sha256": str(row["response_token_sha256"]),
        "counter_deltas": counter_deltas,
        "accepted_by_depth_deltas": accepted_deltas,
        "drafted_by_depth_deltas": drafted_deltas,
    }


def reset_run_caches(runtime: Any, mx: Any) -> dict[str, Any]:
    """Reset cross-run state before the next measurement starts."""

    mx.synchronize()
    clear_hot = getattr(runtime.model, "clear_ngram_hot_cache", None)
    if clear_hot is not None:
        cleared_ngram_rows = int(clear_hot())
    else:
        cleared_ngram_rows = 0
        seen: set[int] = set()
        language_model = getattr(runtime.model, "language_model", runtime.model)
        for layer in getattr(language_model, "layers", ()):
            if "ple" not in layer:
                continue
            sidecar = layer.ple.ple_embedding.ngram_embedding._sidecar
            if sidecar is None or id(sidecar) in seen:
                continue
            seen.add(id(sidecar))
            hot = getattr(sidecar, "_hot", None)
            if hot is not None:
                cleared_ngram_rows += len(hot)
                hot.clear()
    gc.collect()
    return {
        "cleared_ngram_rows": cleared_ngram_rows,
        "mlx_allocator_cache_bytes": int(mx.get_cache_memory()),
        "mlx_allocator_cache_cleared": False,
    }


# --------------------------------------------------------------------------
# Benchmark cells
# --------------------------------------------------------------------------


def build_exact_coding_prompt_ids(
    tokenizer: Any,
    *,
    context: str,
    instruction: str,
    target_tokens: int,
    reasoning_effort: str,
) -> list[int]:
    sentinel = "MTPLX_PR391_CODING_CONTEXT_4E6F91"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(rendered, str) or rendered.count(sentinel) != 1:
        raise RuntimeError("chat template did not preserve the prompt sentinel")
    prefix, suffix = rendered.split(sentinel)
    prefix_ids = list(tokenizer.encode(prefix))
    suffix_ids = list(tokenizer.encode(suffix))
    instruction_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    fixed = len(prefix_ids) + len(instruction_ids) + len(suffix_ids)
    if fixed >= target_tokens:
        raise RuntimeError("coding instruction does not fit the prompt target")
    context_ids = list(tokenizer.encode(context.rstrip() + "\n"))
    if not context_ids:
        raise RuntimeError("coding context encoded to zero tokens")
    budget = target_tokens - fixed
    repeats = (budget + len(context_ids) - 1) // len(context_ids)
    prompt_ids = (
        prefix_ids
        + (context_ids * repeats)[:budget]
        + instruction_ids
        + suffix_ids
    )
    if len(prompt_ids) != target_tokens:
        raise RuntimeError("exact coding prompt construction drifted")
    return prompt_ids


def production_prompt_content() -> str:
    """The exact 16,384-token coding prompt content, pinned by SHA-256."""

    context = (FIXTURES / "qwen38_generation_context.py").read_text()
    case = json.loads(
        (FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
        .read_text()
        .splitlines()[0]
    )
    content = context[:62587] + "\n\n" + case["prompt"]
    if hashlib.sha256(content.encode()).hexdigest() != EXPECTED_PROMPT:
        raise RuntimeError("prompt content drifted")
    return content


def prompt_fixture_sha256() -> dict[str, str]:
    """SHA-256 of both prompt fixtures, for the receipt.

    ``EXPECTED_PROMPT`` only pins the 16,384-token *content* string; a prompt
    built at any other ``--prompt-tokens`` comes from the same two files, so
    the receipt carries their hashes instead of a content hash that would only
    describe the default.
    """

    return {
        name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        for name in (
            "qwen38_generation_context.py",
            "qwen38_naturalistic_generation_patch.jsonl",
        )
    }


def templated_ids(result: Any) -> list[int]:
    """Normalise ``apply_chat_template(tokenize=True)`` across tokenizers.

    The runtime tokenizer returns a flat id list, for which this is a no-op.
    A bare ``transformers`` 5.x tokenizer -- what the pure-Python tests use to
    rebuild the same prompt without MLX -- returns a ``BatchEncoding`` whose
    ``len()`` is the number of KEYS, and either flavour can return a batch of
    one.  Mirrors ``longprompt_agreement_screen.templated_ids``.
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


def build_production_prompt_ids(
    tokenizer: Any, *, prompt_tokens: int = DEFAULT_PROMPT_TOKENS
) -> list[int]:
    """The measured cell's prompt, exactly ``prompt_tokens`` tokens long.

    At the default 16,384 this is the pinned production prompt, built exactly
    as it always was, so existing receipts stay comparable byte for byte.  At
    any other length it is ``build_exact_coding_prompt_ids`` over the same
    SHA-pinned fixture pair -- the driver's own length-targeting path, the one
    ``build_benchmark_cells`` already uses for its 64K and 128K cells -- so
    ``--prompt-tokens 65536`` reproduces ``coding-64k-1k-xhigh-t1``.
    ``production_prompt_content`` is called either way: it is the fixture
    drift gate.
    """

    target = int(prompt_tokens)
    if target not in PROMPT_TOKEN_CHOICES:
        raise ValueError(
            f"--prompt-tokens must be one of "
            f"{', '.join(str(value) for value in PROMPT_TOKEN_CHOICES)}, "
            f"got {target}"
        )
    content = production_prompt_content()
    if target == DEFAULT_PROMPT_TOKENS:
        prompt_ids = templated_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort="xhigh",
            )
        )
    else:
        context = (FIXTURES / "qwen38_generation_context.py").read_text()
        case = json.loads(
            (FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
            .read_text()
            .splitlines()[0]
        )
        prompt_ids = build_exact_coding_prompt_ids(
            tokenizer,
            context=context,
            instruction=case["prompt"],
            target_tokens=target,
            reasoning_effort="xhigh",
        )
    if len(prompt_ids) != target:
        raise RuntimeError(
            f"prompt has {len(prompt_ids)} tokens, expected {target}"
        )
    return prompt_ids


def build_production_cell(
    runtime: Any,
    sampler_type: Any,
    *,
    label: str,
    max_tokens: int,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
) -> dict[str, Any]:
    """The ``coding-16k-1k-xhigh-t1`` cell, constructed identically.

    ``prompt_tokens`` resizes only the prompt; the label, the production-cell
    name, the sampler and the reasoning effort are untouched, so the receipt
    path and the summary table are the same at any length.
    """

    prompt_ids = build_production_prompt_ids(
        runtime.tokenizer, prompt_tokens=prompt_tokens
    )
    return {
        "label": label,
        "production_cell": PRODUCTION_CELL_LABEL,
        "prompt_ids": prompt_ids,
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh",
        "thinking": True,
        "sampler": sampler_type(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ),
        "sampler_receipt": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    }


def graph_warmup_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The unmeasured warm-up copy of ``cell``.

    A shallow copy with a different label and nothing else: the warm-up
    prefills the SAME prompt ids -- so the same ``--prompt-tokens`` length
    and the same chunk layout -- as the run it is warming.  A warm-up at a
    different length would leave the measured first chunk cold, which is
    the entire thing ``--warm-graph`` exists to prevent.
    """

    warm = dict(cell)
    warm["label"] = f"{cell['label']}-unmeasured-graph-warmup"
    return warm


def build_benchmark_cells(runtime: Any, sampler_type: Any) -> list[dict[str, Any]]:
    context = (FIXTURES / "qwen38_generation_context.py").read_text()
    case = json.loads(
        (FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
        .read_text()
        .splitlines()[0]
    )
    vanity_ids = list(
        runtime.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a concise Python function that returns whether "
                        "a string is a palindrome. Return code only."
                    ),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    cells = [
        {
            "label": "vanity-palindrome-t0-thinking-off",
            "prompt_ids": vanity_ids,
            "max_tokens": 100,
            "reasoning_effort": "off",
            "thinking": False,
            "sampler": sampler_type(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            ),
            "sampler_receipt": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        }
    ]
    for reasoning_effort in ("xhigh", "low"):
        for prompt_tokens in (16_384, 65_536, 131_072):
            if prompt_tokens == 16_384 and reasoning_effort == "xhigh":
                prompt_ids = list(
                    runtime.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": (
                                    context[:62587] + "\n\n" + case["prompt"]
                                ),
                            }
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=True,
                        reasoning_effort=reasoning_effort,
                    )
                )
                if len(prompt_ids) != prompt_tokens:
                    raise RuntimeError(
                        "fixed 16K coding prompt token count drifted"
                    )
            else:
                prompt_ids = build_exact_coding_prompt_ids(
                    runtime.tokenizer,
                    context=context,
                    instruction=case["prompt"],
                    target_tokens=prompt_tokens,
                    reasoning_effort=reasoning_effort,
                )
            cells.append(
                {
                    "label": (
                        f"coding-{prompt_tokens // 1024}k-1k-"
                        f"{reasoning_effort}-t1"
                    ),
                    "prompt_ids": prompt_ids,
                    "max_tokens": 1_024,
                    "reasoning_effort": reasoning_effort,
                    "thinking": True,
                    "sampler": sampler_type(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=20,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                    ),
                    "sampler_receipt": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": 0.0,
                        "repetition_penalty": 1.0,
                    },
                }
            )
    return cells


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def check_prompt_tokens(args: argparse.Namespace) -> None:
    """Refuse ``--prompt-tokens`` combinations that would measure nothing.

    Both are fail-closed rather than silently ignored: the matrix carries
    its own per-cell lengths, and the PR391 reference rows were recorded
    against the 16,384-token production prompt, so parity at any other
    length is a guaranteed false drift.
    """

    prompt_tokens = int(getattr(args, "prompt_tokens", DEFAULT_PROMPT_TOKENS))
    if prompt_tokens == DEFAULT_PROMPT_TOKENS:
        return
    if getattr(args, "benchmark_matrix", False):
        raise RuntimeError(
            "--prompt-tokens does not apply to --benchmark-matrix, which "
            "carries its own per-cell prompt lengths"
        )
    if getattr(args, "require_reference_token_parity", False):
        raise RuntimeError(
            "--require-reference-token-parity is only defined at "
            f"--prompt-tokens {DEFAULT_PROMPT_TOKENS}; the reference rows "
            "were recorded against the pinned production prompt"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument(
        "--expected-source",
        default=None,
        help="40-character commit the source tree must be at (optional).",
    )
    parser.add_argument("--expected-file", action="append", default=[])
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Record instead of refusing an uncommitted benchmark source.",
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument(
        "--guard-mode",
        choices=("auto", "attestation", "window"),
        default="auto",
    )
    parser.add_argument(
        "--thermal-gate-max-c",
        type=thermal_gate_max_c,
        default=DEFAULT_THERMAL_MAX_C,
        metavar="CELSIUS",
        help=(
            "Thermal gate threshold in C (default 40.0). The gate is always "
            "on; this only moves the threshold."
        ),
    )
    parser.add_argument(
        "--prewarm-ngram-table",
        action="store_true",
        help=(
            "Read the model n-gram table sequentially once before the timed "
            "cells; sets page_cache_regime=prewarmed."
        ),
    )
    parser.add_argument(
        "--retain-events",
        action="store_true",
        help=(
            "Set MTPLX_DROP_EVENTS=0 so per-cycle arrays are recorded. "
            "Costs host time per cycle; off by default."
        ),
    )
    parser.add_argument(
        "--d3-softfloat64-route",
        action="store_true",
        help=(
            "Install the PR391 exact Metal softfloat64 D3 selector/verifier "
            "route (off by default; the retained 67.818 tok/s control did "
            "not use it)."
        ),
    )
    parser.add_argument(
        "--require-reference-token-parity",
        action="store_true",
        help="Fail the arm when a seed's tokens drift from the PR391 reference.",
    )
    parser.add_argument(
        "--target-mode", choices=("lazy", "batched"), default="batched"
    )
    parser.add_argument("--require-compiled-verify", action="store_true")
    parser.add_argument("--m4-stage3", action="store_true")
    parser.add_argument("--m4-router-top10", action="store_true")
    parser.add_argument("--m4-routed-down", action="store_true")
    parser.add_argument("--m4-down-fused", action="store_true")
    parser.add_argument("--qsa-gather", choices=("0", "1"))
    parser.add_argument("--qsa-fused-kv-gather", action="store_true")
    parser.add_argument("--qsa-direct-attention", action="store_true")
    parser.add_argument("--nax-verify", action="store_true")
    parser.add_argument("--lazy-d3", action="store_true")
    parser.add_argument("--fixed-d3-step", choices=("0", "1"))
    parser.add_argument("--async-draft-submit", choices=("0", "1"))
    parser.add_argument("--fixed-draft-support", choices=("0", "1"))
    parser.add_argument("--compact-frspec", choices=("0", "1"))
    parser.add_argument("--pipelined-mtp-hidden", choices=("0", "1"))
    parser.add_argument("--direct-mtp-dispatch", choices=("0", "1"))
    parser.add_argument("--fused-hc-m4", choices=("0", "1"))
    parser.add_argument("--fixed-m4-hyper", choices=("0", "1"))
    parser.add_argument("--fixed-m4-hyper-tail", choices=("0", "1"))
    parser.add_argument("--compact-ranked-sampler", action="store_true")
    parser.add_argument("--compact-direct-d3", action="store_true")
    parser.add_argument("--stock-d3-host", action="store_true")
    parser.add_argument("--compiled-mtp-prepare", action="store_true")
    parser.add_argument("--relaxed-draft-ties", action="store_true")
    parser.add_argument("--relaxed-target-ties", action="store_true")
    parser.add_argument("--compiled-draft-support", action="store_true")
    parser.add_argument("--compiled-draft-tail", action="store_true")
    parser.add_argument("--compiled-mtp-mlp", action="store_true")
    parser.add_argument("--compiled-routed-mtp-mlp", action="store_true")
    parser.add_argument("--ple-proj-fusion", action="store_true")
    parser.add_argument("--adaptive-dtemp", action="store_true")
    parser.add_argument("--skip-verify-snapshot", action="store_true")
    parser.add_argument("--online-correction-cache", action="store_true")
    parser.add_argument("--prompt-correction-cache", action="store_true")
    parser.add_argument(
        "--correction-cache-key",
        choices=("local_prefix", "source_token", "primary_source"),
        default="local_prefix",
    )
    parser.add_argument("--full-frspec", action="store_true")
    parser.add_argument("--frspec-n", type=int)
    parser.add_argument("--fixed-frspec-template", action="store_true")
    parser.add_argument(
        "--draft-core",
        choices=("stock", "device", "device-lazy"),
        default="stock",
    )
    parser.add_argument("--draft-temperature", type=float, action="append")
    parser.add_argument("--ramp", action="store_true")
    parser.add_argument("--ramp-block", type=int)
    parser.add_argument(
        "--warm-graph",
        action="store_true",
        help=(
            "Run one UNMEASURED copy of every cell first, so the measured run "
            "does not pay the cold first prefill chunk (bimodal ~1.9 s / "
            "~4.4 s on a fresh process). The cold chunk survives in the "
            "receipt as graph_warmup.cells[].first_chunk_cold_s and on each "
            "measured row as first_chunk_cold_s."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        choices=PROMPT_TOKEN_CHOICES,
        default=DEFAULT_PROMPT_TOKENS,
        metavar="N",
        help=(
            "Prompt length of the measured cell, in tokens (default "
            f"{DEFAULT_PROMPT_TOKENS}; one of "
            f"{', '.join(str(value) for value in PROMPT_TOKEN_CHOICES)}). "
            "The default is the pinned production prompt; any other value "
            "is built to exactly N tokens from the same SHA-pinned "
            "fixtures. Label, receipt path and sampler are unchanged. "
            "Refused with --benchmark-matrix, which carries its own "
            "per-cell lengths, and with "
            "--require-reference-token-parity, whose reference rows "
            "were recorded against the pinned production prompt."
        ),
    )
    parser.add_argument("--natural-stop", action="store_true")
    parser.add_argument(
        "--compiled-verify-mode",
        choices=("off", "on", "parity", "parity2"),
        default="on",
    )
    parser.add_argument(
        "--verify-strategy",
        choices=("batched", "capture_commit"),
        default="batched",
    )
    parser.add_argument("--compiled-verify-max-context", type=int)
    parser.add_argument("--context-copy", choices=("0", "1"))
    parser.add_argument("--nan-warning-error", action="store_true")
    parser.add_argument("--benchmark-matrix", action="store_true")
    parser.add_argument("--benchmark-cell", action="append", default=[])
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Construction-time MTPLX_* model-runtime override.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Non-MTPLX process environment applied before the mlx import "
            "(e.g. MLX_MAX_OPS_PER_BUFFER=...)."
        ),
    )
    return parser


#: MTPLX_* settings that are NOT model-runtime overrides and therefore ride the
#: raw ``--env`` process-environment passthrough instead of ``--candidate-env``.
#:
#: ``--candidate-env`` funnels into ``apply_profile_env``, which refuses any key
#: outside ``mtplx.profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS``.  The keys below
#: are read with a bare ``os.environ.get`` at their use site, never through the
#: profile table, so that refusal would reject them even though they are
#: perfectly legitimate knobs:
#:
#:   MTPLX_CONTEXT_COPY_K       mtplx/context_copy.py:context_copy_block_k()
#:                              -- the copy-block cap (default 24)
#:   MTPLX_CONTEXT_COPY_PROBATION_K
#:                              mtplx/context_copy.py:context_copy_probation_k()
#:                              -- the unproven-lane cap (default 8)
#:   MTPLX_SESSION_BANK_MAX_BYTES
#:                              mtplx/engine_session.py -- session bank ceiling
#:
#: This is an ALLOWLIST, not an escape hatch: every other MTPLX_* key on --env
#: still fails loudly, so a typo or a genuine runtime override put on the wrong
#: channel is caught before the model loads rather than being silently ignored.
#: MTPLX_FABLE_* (the diagnostic namespace: census, probes) rides --env by
#: prefix and needs no entry here.
RAW_ENV_MTPLX_READERS = {
    "MTPLX_CONTEXT_COPY_K": "mtplx/context_copy.py:context_copy_block_k",
    "MTPLX_CONTEXT_COPY_PROBATION_K": (
        "mtplx/context_copy.py:context_copy_probation_k"
    ),
    "MTPLX_SESSION_BANK_MAX_BYTES": "mtplx/engine_session.py",
}
RAW_ENV_MTPLX_KEYS = frozenset(RAW_ENV_MTPLX_READERS)


def is_raw_env_mtplx_key(key: str) -> bool:
    """Does this MTPLX_* key belong on the raw ``--env`` passthrough?"""

    return key.startswith("MTPLX_FABLE_") or key in RAW_ENV_MTPLX_KEYS


def parse_key_values(
    settings: list[str], *, flag: str, require_mtplx: bool
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for setting in settings:
        try:
            key, value = setting.split("=", 1)
        except ValueError as exc:
            raise RuntimeError(f"invalid {flag} value: {setting!r}") from exc
        # MTPLX_FABLE_* is the diagnostic namespace (census etc.) and
        # RAW_ENV_MTPLX_KEYS are read straight off os.environ; neither is a
        # model-runtime override, so both ride the raw --env passthrough.
        is_mtplx = key.startswith("MTPLX_") and not is_raw_env_mtplx_key(key)
        if not key or not value or key in parsed:
            raise RuntimeError(f"invalid or duplicate {flag} value: {setting!r}")
        if require_mtplx and not is_mtplx:
            if key.startswith("MTPLX_"):
                raise RuntimeError(
                    f"{key} is a raw process-environment setting; put it on "
                    f"--env, not {flag}: {setting!r}"
                )
            raise RuntimeError(f"{flag} keys must start with MTPLX_: {setting!r}")
        if not require_mtplx and is_mtplx:
            raise RuntimeError(
                f"MTPLX_* keys belong on --candidate-env, not {flag} "
                f"(raw-environment allowlist: "
                f"{', '.join(sorted(RAW_ENV_MTPLX_KEYS))}): {setting!r}"
            )
        parsed[key] = value
    return parsed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_family_overrides(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the construction-time MTPLX overrides for one arm.

    Returns ``(family_overrides, candidate_environment)``.  Extracted from
    ``main`` so the effective-environment contract can be exercised
    without a GPU: every key here must be a member of
    ``mtplx.profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS`` or
    ``apply_profile_env`` refuses the whole arm.
    """

    family_overrides = {
        "MTPLX_SKIP_VERIFY_SNAPSHOT": "1" if args.skip_verify_snapshot else "0",
        "MTPLX_COMPILED_VERIFY": args.compiled_verify_mode,
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS": (
            "1" if args.target_mode == "lazy" else "0"
        ),
        "MTPLX_BATCH_TARGET_ARRAYS": (
            "0" if args.target_mode == "lazy" else "1"
        ),
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
        "MTPLX_NAX_VERIFY": "1" if args.nax_verify else "0",
    }
    if args.retain_events:
        # Route it through the family overrides so the effective-environment
        # drift check below still compares equal.
        family_overrides["MTPLX_DROP_EVENTS"] = "0"
    if args.qsa_gather is not None:
        family_overrides["MTPLX_QSA_GATHER"] = args.qsa_gather
    if args.qsa_fused_kv_gather:
        family_overrides["MTPLX_QSA_M4_FUSED_KV_GATHER"] = "1"
    if args.qsa_direct_attention:
        family_overrides["MTPLX_QSA_M4_DIRECT_ATTENTION"] = "1"
    if args.require_compiled_verify:
        family_overrides["MTPLX_QWEN4_FIXED_M4_VERIFY"] = "1"
    if args.compiled_mtp_prepare:
        family_overrides["MTPLX_QWEN4_COMPILED_MTP_PREPARE"] = "1"
    if args.relaxed_draft_ties:
        family_overrides["MTPLX_QWEN4_RELAXED_DRAFT_TIES"] = "1"
    if args.compiled_draft_support:
        family_overrides["MTPLX_QWEN4_COMPILED_DRAFT_SUPPORT"] = "1"
    if args.compiled_draft_tail:
        family_overrides["MTPLX_QWEN4_COMPILED_DRAFT_TAIL"] = "1"
    if args.ple_proj_fusion:
        family_overrides["MTPLX_FUSE_PROJ"] = "ple"
    if args.relaxed_target_ties:
        family_overrides["MTPLX_QWEN4_RELAXED_TARGET_TIES"] = "1"
    if args.compiled_mtp_mlp:
        family_overrides["MTPLX_QWEN4_COMPILED_MTP_MLP"] = "1"
    if args.compiled_routed_mtp_mlp:
        family_overrides["MTPLX_QWEN4_COMPILED_ROUTED_MTP_MLP"] = "1"
    if args.lazy_d3:
        family_overrides["MTPLX_QWEN4_FIXED_D3_DRAFT"] = "1"
    if args.m4_stage3:
        family_overrides["MTPLX_QWEN4_M4_STAGE3"] = "1"
    if args.m4_router_top10:
        family_overrides["MTPLX_QWEN4_M4_ROUTER_TOP10"] = "1"
    if args.m4_routed_down:
        family_overrides["MTPLX_QWEN4_M4_ROUTED_DOWN_COMBINE"] = "1"
    if args.fixed_d3_step is not None:
        family_overrides["MTPLX_QWEN4_FIXED_D3_STEP"] = args.fixed_d3_step
    if args.async_draft_submit is not None:
        family_overrides["MTPLX_QWEN4_ASYNC_DRAFT_SUBMIT"] = args.async_draft_submit
    if args.fixed_draft_support is not None:
        family_overrides["MTPLX_QWEN4_FIXED_DRAFT_SUPPORT"] = (
            args.fixed_draft_support
        )
    if args.compact_frspec is not None:
        family_overrides["MTPLX_QWEN4_COMPACT_FRSPEC"] = args.compact_frspec
    if args.pipelined_mtp_hidden is not None:
        family_overrides["MTPLX_QWEN4_PIPELINED_MTP_HIDDEN"] = (
            args.pipelined_mtp_hidden
        )
    if args.direct_mtp_dispatch is not None:
        family_overrides["MTPLX_QWEN4_DIRECT_MTP_DISPATCH"] = (
            args.direct_mtp_dispatch
        )

    candidate_environment = parse_key_values(
        args.candidate_env, flag="--candidate-env", require_mtplx=True
    )
    family_overrides.update(candidate_environment)
    return family_overrides, candidate_environment


def main() -> int:
    args = build_parser().parse_args()
    check_prompt_tokens(args)
    if args.nan_warning_error:
        warnings.filterwarnings(
            "error",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )

    guard = acquire_guard(args.guard_mode)
    source_path = args.source.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    source_receipt = verify_inputs(
        source_path,
        args.expected_source,
        args.expected_file,
        allow_dirty_source=args.allow_dirty_source,
        model_path=model_path,
    )
    available = wait_for_memory()
    print(
        f"[fable-abba] lock attested ({guard['mode']}); "
        f"reclaimable={available / 1024**3:.2f} GiB",
        flush=True,
    )
    sys.path.insert(0, str(source_path))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    extra_environment = parse_key_values(
        args.env, flag="--env", require_mtplx=False
    )
    os.environ.update(extra_environment)
    # ONE receipt field for everything the process was told through --env.
    #
    # Two failures forced it, and they need different halves of the same
    # record. (a) The 2026-09-01 PLE-lookahead window armed its candidate
    # through --candidate-extra-env, which lands here and NOT in
    # `candidate_environment`; both arms' receipts looked identical and an
    # inert lane read as a 2 s regression -- so EVERY raw setting must appear,
    # not just the validated overrides. (b) The allowlisted MTPLX_* settings
    # (RAW_ENV_MTPLX_READERS) change MTPLX behaviour, so their requested value
    # has to be shown next to the value actually in force: a difference means
    # something later in setup overwrote the key, which is precisely the
    # failure this field exists to expose.
    #
    # `reader` is None for keys outside the allowlist (MLX_*, MTPLX_FABLE_*),
    # which is also how a reader tells the two classes apart.
    globals()["_EXTRA_ENVIRONMENT"] = dict(extra_environment)
    process_environment_overrides = {
        key: {
            "requested": value,
            "effective": os.environ.get(key),
            "reader": RAW_ENV_MTPLX_READERS.get(key),
        }
        for key, value in sorted(_EXTRA_ENVIRONMENT.items())
    }

    family_overrides, candidate_environment = build_family_overrides(args)

    mtplx_file = assert_mtplx_tree(source_path)
    from mtplx.profiles import apply_profile_env, get_profile

    expected_environment = get_profile("turbo").env_dict()
    expected_environment.update(family_overrides)
    if args.context_copy is not None:
        expected_environment["MTPLX_CONTEXT_COPY"] = args.context_copy
    if args.compiled_verify_max_context is not None:
        expected_environment["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"] = str(
            args.compiled_verify_max_context
        )
    for key in expected_environment:
        os.environ.pop(key, None)
    apply_profile_env("turbo", runtime_env_overrides=family_overrides)
    if args.context_copy is not None:
        os.environ["MTPLX_CONTEXT_COPY"] = args.context_copy
    os.environ.pop("MTPLX_QWEN4_M4_DOWN_FUSED", None)
    if args.m4_down_fused:
        os.environ["MTPLX_QWEN4_M4_DOWN_FUSED"] = "1"
    if args.compiled_verify_max_context is not None:
        os.environ["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"] = str(
            args.compiled_verify_max_context
        )
    if args.compact_direct_d3:
        os.environ["MTPLX_QWEN4_COMPACT_DIRECT_D3"] = "1"
    if args.fused_hc_m4 == "1":
        os.environ["MTPLX_FUSED_HC"] = "1"
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "MTPLX_CONTEXT_WINDOW_TOKENS": "262144",
            "MTPLX_NGRAM_HOT_MB": "1024",
            "MTPLX_MEMORY_LIMIT_BYTES": str(MEMORY_LIMIT_BYTES),
            "MTPLX_WIRED_LIMIT_BYTES": str(WIRED_LIMIT_BYTES),
            "MTPLX_ADAPTIVE_DTEMP": "1" if args.adaptive_dtemp else "0",
            "MTPLX_STATE_REBASE_EVERY": "0",
            "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "0",
        }
    )
    if args.ramp:
        os.environ["MTPLX_RAMP_ENABLED"] = "1"
    if args.ramp_block is not None:
        os.environ["MTPLX_RAMP_BLOCK"] = str(args.ramp_block)
    if args.compact_ranked_sampler:
        os.environ["MTPLX_QWEN4_COMPACT_RANKED_SAMPLER"] = "1"
    if args.stock_d3_host:
        os.environ["MTPLX_QWEN4_STOCK_D3_HOST"] = "1"
    if args.fixed_frspec_template:
        os.environ["MTPLX_FRSPEC_FIXED_B1S1"] = "1"
    if (
        args.full_frspec
        or args.compact_direct_d3
        or args.compact_frspec == "1"
        or args.pipelined_mtp_hidden == "1"
        or args.direct_mtp_dispatch == "1"
    ):
        os.environ.update(
            {
                "MTPLX_FRSPEC_DRAFT": "1",
                "MTPLX_FRSPEC_VOCAB": (
                    str(source_path / "mtplx/data/qwen38_code_ranked_64k.json")
                    if args.frspec_n is not None
                    else "builtin:qwen38-code-64k"
                ),
            }
        )
    if args.frspec_n is not None:
        if not args.full_frspec:
            raise RuntimeError("--frspec-n requires --full-frspec")
        os.environ["MTPLX_FRSPEC_N"] = str(args.frspec_n)
    observed = {key: os.environ.get(key) for key in expected_environment}
    if observed != expected_environment:
        drift = {
            key: (expected_environment[key], observed[key])
            for key in expected_environment
            if observed[key] != expected_environment[key]
        }
        raise RuntimeError(f"effective construction environment drifted: {drift}")
    fixed_d3_exclusions = {
        key: os.environ.get(key)
        for key in (
            "MTPLX_ADAPTIVE_DTEMP",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD",
        )
    }
    expected_fixed_d3_exclusions = {
        "MTPLX_ADAPTIVE_DTEMP": "1" if args.adaptive_dtemp else "0",
        "MTPLX_STATE_REBASE_EVERY": "0",
        "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "0",
    }
    if fixed_d3_exclusions != expected_fixed_d3_exclusions:
        raise RuntimeError(
            f"fixed-D3 replacement modes were not disabled: {fixed_d3_exclusions}"
        )

    from mtplx.server.openai import _apply_metal_memory_caps

    memory_caps = _apply_metal_memory_caps(
        minimum_resident_bytes=MINIMUM_RESIDENT_BYTES
    )
    if not memory_caps.get("applied"):
        raise RuntimeError(f"Metal memory caps did not apply: {memory_caps}")

    import mlx.core as mx

    mx.reset_peak_memory()

    from mtplx.draft_lm_head import _install_draft_lm_head
    from mtplx.fable_indexer_reuse import (
        indexer_reuse_counters,
        reset_indexer_reuse_counters,
    )
    from mtplx.generation import generate_mtpk
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    runtime = load(model_path, mtp=True)

    # ---- construction-time route assertions (as reviewed) -----------------
    # The reviewed driver's GDN g/beta, GDN norm-gate and QSA flat-score
    # assertion blocks are omitted: their env keys
    # (MTPLX_QWEN4_GDN_GBETA_M4, MTPLX_QWEN4_GDN_NORM_GATE_M4,
    # MTPLX_QWEN4_QSA_M4_FLAT_SCORES) are not in this branch's
    # MODEL_RUNTIME_ENV_OVERRIDE_KEYS, so apply_profile_env refuses them and
    # the blocks could never fire. Re-add them alongside the keys.
    ple_prefix_reuse_expected = (
        candidate_environment.get("MTPLX_QWEN4_M4_PLE_PREFIX_REUSE", "0") == "1"
    )
    ple_prefix_reuse_observed = bool(
        getattr(runtime, "qwen4_fixed_m4_ple_prefix_reuse", False)
    )
    if ple_prefix_reuse_observed != ple_prefix_reuse_expected:
        raise RuntimeError(
            "fixed-M4 PLE prefix-reuse route did not match construction flag: "
            f"expected={ple_prefix_reuse_expected}, "
            f"observed={ple_prefix_reuse_observed}"
        )
    ple_m23_direct_expected = (
        candidate_environment.get("MTPLX_QWEN4_M4_PLE_M23_DIRECT", "0") == "1"
    )
    ple_m23_direct_observed = bool(
        getattr(runtime, "qwen4_fixed_m4_ple_m23_direct", False)
    )
    if ple_m23_direct_observed != ple_m23_direct_expected:
        raise RuntimeError(
            "fixed-M4 PLE M2/M3 direct route did not match construction flag: "
            f"expected={ple_m23_direct_expected}, "
            f"observed={ple_m23_direct_observed}"
        )
    gdn_prefix_states_expected = (
        candidate_environment.get("MTPLX_QWEN4_M4_GDN_PREFIX_STATES", "0") == "1"
    )
    gdn_prefix_states_observed = bool(
        getattr(runtime, "qwen4_fixed_m4_gdn_prefix_states", False)
    )
    if gdn_prefix_states_observed != gdn_prefix_states_expected:
        raise RuntimeError(
            "fixed-M4 GDN prefix-state route did not match construction flag: "
            f"expected={gdn_prefix_states_expected}, "
            f"observed={gdn_prefix_states_observed}"
        )
    if args.m4_stage3:
        m4_report = getattr(runtime, "qwen4_m4_stage3_report", None) or {}
        residual_tail_enabled = (
            os.environ.get("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL") == "1"
        )
        routed_glu_enabled = os.environ.get("MTPLX_QWEN4_M4_ROUTED_GLU") == "1"
        expected_boundary = (
            "paired_routed_q4g32_glu_reduce_shared_add_mlp_residual"
            if routed_glu_enabled
            else "routed_q4g32_reduce_shared_add_mlp_residual"
            if residual_tail_enabled
            else (
                "materialized_routed_q4_down_then_stock_combine"
                if args.m4_down_fused
                else "stock_qmm_combine_tail"
            )
        )
        if m4_report.get("boundary") != expected_boundary:
            raise RuntimeError(
                f"M4 down route drifted: expected {expected_boundary}, "
                f"got {m4_report}"
            )
        if residual_tail_enabled and (
            int(m4_report.get("exact_layers", 0)) != 48
            or int(m4_report.get("combined_residual_tail_layers", 0)) != 48
            or m4_report.get("reference_boundary")
            != "retained_m4_routed_down_then_stock_residual"
        ):
            raise RuntimeError(f"M4 residual-tail ownership drifted: {m4_report}")
        if routed_glu_enabled and (
            m4_report.get("paired_routed_glu") is not True
            or int(m4_report.get("paired_routed_glu_layers", 0)) != 48
        ):
            raise RuntimeError(
                f"M4 paired routed-GLU ownership drifted: {m4_report}"
            )
        print(
            "[fable-abba] M4 route " + json.dumps(m4_report, sort_keys=True),
            flush=True,
        )
    if args.relaxed_draft_ties and not getattr(
        runtime, "qwen4_relaxed_draft_ties", False
    ):
        raise RuntimeError("relaxed draft-tie route did not install")
    if args.compiled_draft_support:
        report = getattr(runtime, "qwen4_compiled_relaxed_draft_report", None) or {}
        if report.get("installed") is not True:
            raise RuntimeError(f"compiled draft support did not install: {report}")
    if args.relaxed_target_ties and not getattr(
        runtime, "qwen4_relaxed_target_ties", False
    ):
        raise RuntimeError("relaxed target-tie route did not install")
    if args.compiled_mtp_prepare:
        report = getattr(runtime, "qwen4_compiled_mtp_prepare_report", None) or {}
        if report.get("installed") is not True:
            raise RuntimeError(f"compiled MTP prepare did not install: {report}")
    if args.compiled_mtp_mlp:
        report = getattr(runtime, "qwen4_compiled_mtp_mlp_report", None) or {}
        if report.get("installed") is not True:
            raise RuntimeError(f"compiled MTP MLP did not install: {report}")
    if args.compiled_routed_mtp_mlp:
        report = (
            getattr(runtime, "qwen4_compiled_routed_mtp_mlp_report", None) or {}
        )
        if report.get("installed") is not True:
            raise RuntimeError(f"compiled routed MTP MLP did not install: {report}")
    if args.stock_d3_host and not runtime.qwen4_stock_d3_host:
        raise RuntimeError("stock-D3 host route did not install")
    if args.fixed_m4_hyper == "1":
        report = getattr(runtime, "qwen4_fixed_verify_report", None) or {}
        if report.get("hyper_modules") != 96:
            raise RuntimeError(f"fixed-M4 hyper route did not install: {report}")
    if args.fixed_m4_hyper_tail == "1":
        report = getattr(runtime, "qwen4_fixed_verify_report", None) or {}
        if report.get("hyper_tail_modules") != 96:
            raise RuntimeError(f"fixed-M4 hyper tail did not install: {report}")
    if args.m4_routed_down and (
        (getattr(runtime, "qwen4_m4_stage3_report", None) or {}).get("boundary")
        != "routed_down_q4g32_combine"
    ):
        raise RuntimeError("fixed-M4 routed-down combine did not install")
    if args.m4_router_top10 and (
        (getattr(runtime, "qwen4_m4_stage3_report", None) or {}).get("router")
        != "row_owned_m4_top10"
    ):
        raise RuntimeError("fixed-M4 row-owned top-10 router did not install")
    if args.fixed_d3_step == "1" and not getattr(
        runtime, "qwen4_fixed_d3_step", False
    ):
        raise RuntimeError("fixed-D3 cached draft step did not install")
    if args.fixed_draft_support == "1" and not getattr(
        runtime, "qwen4_fixed_draft_support", False
    ):
        raise RuntimeError("fixed draft support did not install")

    draft_head = _install_draft_lm_head(
        runtime, bits=4, group_size=64, mode="affine"
    )
    if int((draft_head.get("draft_only") or {}).get("bits") or 0) != 4:
        raise RuntimeError("production q4 draft head was not installed")
    if args.compiled_draft_tail:
        compiled_tail = (draft_head.get("frspec") or {}).get("compiled_tail") or {}
        if compiled_tail.get("installed") is not True:
            raise RuntimeError(f"compiled draft tail did not install: {draft_head}")
    if args.compact_ranked_sampler and (
        (draft_head.get("compact_ranked_sampler") or {}).get("installed") is not True
    ):
        raise RuntimeError("compact ranked sampler did not install")
    if args.compact_direct_d3 and not getattr(
        runtime, "qwen4_compact_direct_d3", False
    ):
        raise RuntimeError("compact direct D3 route did not install")
    if args.full_frspec:
        text_model = getattr(runtime.model, "language_model", runtime.model)
        frspec_ids = getattr(text_model, "_mtplx_frspec_ids", None)
        expected_frspec_n = args.frspec_n or 65_536
        if frspec_ids is None or tuple(frspec_ids.shape) != (expected_frspec_n,):
            raise RuntimeError("full-padded FRSpec head did not install")
        frspec_report = draft_head.get("frspec") or {}
        if args.fixed_frspec_template and (
            frspec_report.get("scatter_template") != "fixed_b1s1"
        ):
            raise RuntimeError(
                f"fixed B1/S1 FRSpec template did not install: {frspec_report}"
            )

    # ---- PR391 D3 softfloat64 route prebind (ported injection) ------------
    d3_route_module = None
    d3_prebound = None
    d3_verifier = None
    d3_verifier_prewarm = None
    if args.d3_softfloat64_route:
        import mtplx.generation as d3_generation_module
        from mtplx.kernels.pr391_softfloat64_verifier_decision import (
            bind_pr391_softfloat64_verifier_decision,
        )
        from scripts.pr391_metal_choice_benchmark_launcher import (
            prebind_softfloat64_choice_kernel,
            prewarm_softfloat64_verifier_decision,
        )

        d3_route_module = d3_generation_module
        d3_prebound = prebind_softfloat64_choice_kernel()
        d3_verifier = bind_pr391_softfloat64_verifier_decision()
        d3_verifier_prewarm = prewarm_softfloat64_verifier_decision(
            d3_prebound.mx, d3_verifier
        )
        print(
            "[fable-abba] D3 softfloat64 route prebound "
            + json.dumps(dict(d3_verifier_prewarm), sort_keys=True),
            flush=True,
        )

    # ---- cells ------------------------------------------------------------
    if args.benchmark_matrix:
        cells = build_benchmark_cells(runtime, SamplerConfig)
        if args.benchmark_cell:
            requested_cells = set(args.benchmark_cell)
            known_cells = {str(cell["label"]) for cell in cells}
            unknown_cells = sorted(requested_cells - known_cells)
            if unknown_cells:
                raise RuntimeError(
                    f"unknown benchmark cells: {', '.join(unknown_cells)}"
                )
            cells = [
                cell for cell in cells if str(cell["label"]) in requested_cells
            ]
    else:
        base_cell = build_production_cell(
            runtime,
            SamplerConfig,
            label=args.label,
            max_tokens=args.max_tokens,
            prompt_tokens=args.prompt_tokens,
        )
        cells = [base_cell]
        if args.draft_temperature:
            cells = []
            for draft_temperature in args.draft_temperature:
                cell = dict(base_cell)
                cell["label"] = f"{args.label}-draft-t{draft_temperature:g}"
                cell["draft_sampler"] = SamplerConfig(
                    temperature=draft_temperature,
                    top_p=0.95,
                    top_k=20,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                )
                cells.append(cell)

    ngram_prewarm: dict[str, Any] | None = None
    if args.prewarm_ngram_table:
        ngram_prewarm = prewarm_ngram_table(model_path / NGRAM_TABLE_NAME)
        print(
            "[fable-abba] n-gram prewarm "
            + json.dumps(ngram_prewarm, sort_keys=True),
            flush=True,
        )

    after_load_memory = {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }

    def run(
        cell: dict[str, Any],
        sequence: int,
        seed: int,
        *,
        cold_first_chunk_s: float | None = None,
    ) -> dict[str, Any]:
        thermal_receipt = wait_for_temperature(args.thermal_gate_max_c)
        reset_receipt = reset_run_caches(runtime, mx)
        # Per-seed, so a shortfall against (depth-1) * cycles is attributable
        # to the run that produced it rather than smeared over the arm.
        reset_indexer_reuse_counters()
        mx.reset_peak_memory()
        prompt_ids = cell["prompt_ids"]
        max_tokens = int(cell["max_tokens"])
        sampler = cell["sampler"]
        draft_sampler = cell.get("draft_sampler", sampler)

        # PR391 D3 softfloat64 route install (ported injection: this is the
        # block transform_metal_choice_driver splices in immediately before
        # `started = time.perf_counter()`).
        d3_route = None
        if args.d3_softfloat64_route:
            from scripts.pr391_metal_choice_benchmark_launcher import (
                PR391DirectSoftFloat64D3Route,
            )

            d3_route = PR391DirectSoftFloat64D3Route.install(
                d3_route_module,
                expected_seed=seed,
                max_output_tokens=max_tokens,
                kernel_module=d3_prebound,
                verifier_kernel=d3_verifier,
                verifier_prewarm=d3_verifier_prewarm,
                target_sampler=sampler,
                draft_sampler=draft_sampler,
            )

        started = time.perf_counter()
        try:
            output = generate_mtpk(
                runtime,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_sampler=draft_sampler,
                speculative_depth=3,
                seed=seed,
                stop_token_ids=None if args.natural_stop else set(),
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy=args.verify_strategy,
                verify_core="linear-gdn-from-conv-tape",
                draft_core=args.draft_core,
                online_correction_cache=args.online_correction_cache,
                online_correction_cache_key=args.correction_cache_key,
                prompt_correction_cache=args.prompt_correction_cache,
            )
        except BaseException:
            if d3_route is not None:
                d3_route.close()
            raise
        if args.pipelined_mtp_hidden == "1" and not getattr(
            runtime, "qwen4_pipelined_mtp_hidden_request", False
        ):
            raise RuntimeError("pipelined MTP hidden route did not engage")
        if args.direct_mtp_dispatch == "1" and not getattr(
            runtime, "qwen4_direct_mtp_dispatch_request", False
        ):
            raise RuntimeError("direct MTP dispatch route did not engage")
        if args.draft_core in {"device", "device-lazy"}:
            device = dict(output.stats.draft_core or {})
            if int(device.get("device_calls", 0)) <= 0:
                raise RuntimeError(f"device draft core did not engage: {device}")
            if int(device.get("device_fallbacks", -1)) != 0:
                raise RuntimeError(f"device draft core fell back: {device}")
        if args.compact_direct_d3 and int(output.stats.mtp_forward_calls) != 0:
            raise RuntimeError(
                "compact direct D3 did not bypass runtime draft dispatch"
            )
        if args.fixed_d3_step == "1":
            fixed_d3_bank = getattr(runtime, "_qwen4_fixed_d3_step_bank", None)
            if not isinstance(fixed_d3_bank, dict) or fixed_d3_bank.get(
                "rng_mode"
            ) != "host-integer-threshold":
                raise RuntimeError("fixed-D3 graph did not engage")

        row = stats_receipt(
            output,
            str(cell["label"]),
            sequence,
            seed,
            time.perf_counter() - started,
        )
        row["pre_run_reset"] = reset_receipt
        # MTPLX_FABLE_INDEXER_REUSE engagement. On an unarmed arm both are 0,
        # which is what "the control really was the control" looks like.
        row["indexer_reuse"] = {
            "armed": os.environ.get("MTPLX_FABLE_INDEXER_REUSE") == "1",
            **indexer_reuse_counters(),
        }

        # PR391 D3 attach block (ported injection).
        if d3_route is not None:
            from scripts.pr391_metal_choice_benchmark_launcher import (
                build_exact_output_parity_receipt,
                validate_metal_choice_receipt,
            )

            d3_route.close()
            d3_receipt = d3_route.finish_receipt(stats=output.stats)
            validate_metal_choice_receipt(d3_receipt)
            row["metal_softfloat64_choice_route"] = d3_receipt
            row["response_token_ids"] = [int(token) for token in output.tokens]
            row["softfloat64_output_parity"] = build_exact_output_parity_receipt(row)
            print(
                "[fable-abba] softfloat64 "
                + json.dumps(
                    {
                        "seed": int(row["seed"]),
                        "decode_elapsed_s": float(row["decode_elapsed_s"]),
                        "decode_tok_s": float(row["decode_tok_s"]),
                        "parity": row["softfloat64_output_parity"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not row["softfloat64_output_parity"]["exact_reference_match"]:
                failure_path = OUT_DIR / (
                    f"softfloat64-parity-failure-{sequence}-{seed}.json"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    json.dumps(
                        {
                            "seed": int(seed),
                            "sequence": int(sequence),
                            "response_token_ids": row["response_token_ids"],
                            "generation_events": list(output.stats.events),
                            "parity": row["softfloat64_output_parity"],
                        },
                        sort_keys=True,
                    )
                )
                print(
                    f"[fable-abba] parity diagnostic {failure_path}", flush=True
                )
                raise RuntimeError("Metal softfloat64 output or PCG64 parity failed")

        from scripts.pr391_metal_choice_benchmark_launcher import (
            build_hit_miss_receipt,
        )

        try:
            # Every rate in this receipt divides by a drafted/resolved count;
            # a degenerate cell (no drafts, no resolved verifies) must not
            # destroy an otherwise good measurement.
            row["hit_miss"] = build_hit_miss_receipt(row)
        except ZeroDivisionError as error:
            row["hit_miss"] = {"available": False, "reason": repr(error)}
        row["reference_token_parity"] = reference_token_parity(row)
        row["per_cycle"] = per_cycle_receipt(output.stats)
        row["ple_hot_rows"] = ple_hot_rows_receipt(runtime)
        row["prefill_chunks"] = prefill_chunks_receipt()
        compiled = row["compiled_verify"]
        row["compiled_m4_calls"] = int(compiled.get("compiled_calls", 0))
        row["configured_max_tokens"] = max_tokens
        row["finish_reason"] = output.finish_reason
        row["natural_stop"] = bool(args.natural_stop)
        row["thermal_gate"] = thermal_receipt
        row["prompt_tokens"] = len(prompt_ids)
        row["reasoning_effort"] = cell["reasoning_effort"]
        row["thinking"] = bool(cell["thinking"])
        row["sampler"] = dict(cell["sampler_receipt"])
        row["peak_memory_bytes"] = int(mx.get_peak_memory())
        row["active_memory_bytes"] = int(mx.get_active_memory())
        row["mlx_cache_memory_bytes"] = int(mx.get_cache_memory())
        row["page_cache_regime"] = (
            "prewarmed" if args.prewarm_ngram_table else "as-found"
        )
        # None on a run that was itself the cold one (no --warm-graph, or the
        # warm-up cell): then row["prefill_chunks"][0] IS the cold chunk.
        row["first_chunk_cold_s"] = cold_first_chunk_s
        decoded = runtime.tokenizer.decode(output.tokens)
        row["response_text_chars"] = len(decoded)
        row["response_text_head"] = decoded[:600]
        row["response_text_tail"] = decoded[-600:]
        if args.fixed_d3_step == "1":
            row["fixed_d3_graph"] = {
                "compile_weight_l1_units": list(
                    fixed_d3_bank["compile_weight_l1_units"]
                ),
                "prob_scale": 1 << 24,
            }
        if not args.natural_stop and row["generated_tokens"] != max_tokens:
            raise RuntimeError(f"short production output: {row['generated_tokens']}")
        if args.natural_stop and not (0 < row["generated_tokens"] <= max_tokens):
            raise RuntimeError(
                f"invalid variable production output: {row['generated_tokens']}"
            )
        if args.require_compiled_verify:
            if int(compiled.get("fallback_calls", -1)) != 0:
                raise RuntimeError(f"compiled verifier fell back: {compiled}")
            if int(compiled.get("compiled_calls", 0)) <= 0:
                raise RuntimeError(f"compiled verifier did not engage: {compiled}")
        if row["repair_time_s"] != 0.0:
            raise RuntimeError(f"repair path engaged: {row['repair_time_s']}")
        if (
            args.require_reference_token_parity
            and row["reference_token_parity"]["status"] != "match"
        ):
            raise RuntimeError(
                "reference token parity failed: "
                + json.dumps(row["reference_token_parity"], sort_keys=True)
            )
        print("[fable-abba] " + json.dumps(row, sort_keys=True), flush=True)
        del output
        gc.collect()
        return row

    rows = []
    graph_warmup_cells: list[dict[str, Any]] = []
    sequence = args.sequence
    for cell in cells:
        cold_s: float | None = None
        if args.warm_graph:
            warm_cell = graph_warmup_cell(cell)
            warm_row = run(warm_cell, sequence - 1, args.seed[0])
            cold_s = first_chunk_cold_s(warm_row)
            graph_warmup_cells.append(
                {
                    "label": warm_cell["label"],
                    "measured_label": cell["label"],
                    "first_chunk_cold_s": cold_s,
                    "prefill_chunks": warm_row.get("prefill_chunks") or [],
                    "prompt_eval_time_s": warm_row.get("prompt_eval_time_s"),
                    "ttft_s": warm_row.get("ttft_s"),
                }
            )
        for seed in args.seed:
            rows.append(run(cell, sequence, seed, cold_first_chunk_s=cold_s))
            sequence += 1
    after_run_memory = {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    cell_receipts = []
    for cell in cells:
        cell_rows = [row for row in rows if row["arm"] == cell["label"]]
        if not cell_rows:
            continue
        cell_receipts.append(
            {
                "label": cell["label"],
                "prompt_tokens": len(cell["prompt_ids"]),
                "configured_max_tokens": int(cell["max_tokens"]),
                "generated_tokens": [
                    int(row["generated_tokens"]) for row in cell_rows
                ],
                "finish_reasons": [row["finish_reason"] for row in cell_rows],
                "reasoning_effort": cell["reasoning_effort"],
                "thinking": bool(cell["thinking"]),
                "sampler": dict(cell["sampler_receipt"]),
                "draft_temperature": float(
                    cell.get("draft_sampler", cell["sampler"]).temperature
                ),
                "runs": len(cell_rows),
                "mean_prefill_tok_s": statistics.fmean(
                    row["prefill_tok_s"] for row in cell_rows
                ),
                "mean_decode_tok_s": statistics.fmean(
                    row["decode_tok_s"] for row in cell_rows
                ),
                "mean_wall_s": statistics.fmean(row["wall_s"] for row in cell_rows),
                "mean_ttft_s": statistics.fmean(row["ttft_s"] for row in cell_rows),
                "max_peak_memory_bytes": max(
                    row["peak_memory_bytes"] for row in cell_rows
                ),
            }
        )

    payload = {
        "schema": "mtplx-fable-abba-arm-v1",
        "status": "arm_measured",
        "label": args.label,
        "source_commit": source_receipt["observed_commit"],
        "source_path": str(source_path),
        "source": source_receipt,
        "mtplx_module": mtplx_file,
        "candidate_files": source_receipt["candidate_files"],
        "model_path": str(model_path),
        "model_revision": EXPECTED_MODEL,
        # Only the default length has pinned prompt BYTES; at any other
        # length the identity of the prompt is the fixture pair plus N.
        "prompt_content_sha256": (
            EXPECTED_PROMPT
            if not args.benchmark_matrix
            and args.prompt_tokens == DEFAULT_PROMPT_TOKENS
            else None
        ),
        "prompt_tokens": (
            None if args.benchmark_matrix else int(args.prompt_tokens)
        ),
        "prompt_fixture_sha256": prompt_fixture_sha256(),
        "production_cell": (
            None if args.benchmark_matrix else PRODUCTION_CELL_LABEL
        ),
        "benchmark_cells": cell_receipts,
        "seeds": args.seed,
        "between_run_policy": {
            "thermal_max_c": args.thermal_gate_max_c,
            "fresh_logical_kv": True,
            "mlx_allocator_cache_cleared": False,
            "ngram_hot_cache_mib": 1024,
            "ngram_hot_cache_cleared": True,
            "file_page_cache_cleared": False,
        },
        "page_cache_regime": (
            "prewarmed" if args.prewarm_ngram_table else "as-found"
        ),
        "ngram_prewarm": ngram_prewarm,
        "events_retained": bool(args.retain_events),
        "d3_softfloat64_route": bool(args.d3_softfloat64_route),
        "d3_softfloat64_verifier_prewarm": (
            dict(d3_verifier_prewarm) if d3_verifier_prewarm else None
        ),
        "ngram_hot_mib": 1024,
        "target_distribution_mode": args.target_mode,
        "compiled_verify_mode": args.compiled_verify_mode,
        "verify_strategy": args.verify_strategy,
        "lazy_d3": bool(args.lazy_d3),
        "m4_stage3": bool(args.m4_stage3),
        "m4_router_top10": bool(args.m4_router_top10),
        "qsa_gather": family_overrides["MTPLX_QSA_GATHER"] == "1",
        "qsa_fused_kv_gather": bool(args.qsa_fused_kv_gather),
        "qsa_direct_attention": bool(args.qsa_direct_attention),
        "nax_verify": bool(args.nax_verify),
        "compact_ranked_sampler": bool(args.compact_ranked_sampler),
        "m4_routed_down": bool(args.m4_routed_down),
        "m4_down_fused": bool(args.m4_down_fused),
        "fixed_d3_step": args.fixed_d3_step == "1",
        "async_draft_submit": args.async_draft_submit == "1",
        "fixed_draft_support": args.fixed_draft_support == "1",
        "compact_frspec": args.compact_frspec == "1",
        "pipelined_mtp_hidden": args.pipelined_mtp_hidden == "1",
        "direct_mtp_dispatch": args.direct_mtp_dispatch == "1",
        "fused_hc_m4": args.fused_hc_m4 == "1",
        "fixed_m4_hyper": args.fixed_m4_hyper == "1",
        "fixed_m4_hyper_tail": args.fixed_m4_hyper_tail == "1",
        "draft_core": args.draft_core,
        "full_frspec": bool(args.full_frspec),
        "frspec_n": args.frspec_n or 65_536,
        "fixed_frspec_template": bool(args.fixed_frspec_template),
        "compact_direct_d3": bool(args.compact_direct_d3),
        "stock_d3_host": bool(args.stock_d3_host),
        "compiled_mtp_prepare": bool(args.compiled_mtp_prepare),
        "compiled_draft_tail": bool(args.compiled_draft_tail),
        "ple_proj_fusion": bool(args.ple_proj_fusion),
        "relaxed_draft_ties": bool(args.relaxed_draft_ties),
        "relaxed_target_ties": bool(args.relaxed_target_ties),
        "compiled_mtp_mlp": bool(args.compiled_mtp_mlp),
        "compiled_routed_mtp_mlp": bool(args.compiled_routed_mtp_mlp),
        "graph_warmed_before_measurement": bool(args.warm_graph),
        # The unmeasured warm-up run's own first chunk: the cold number, kept
        # for TTFT-after-restart tracking now that the measured run is warm.
        "graph_warmup": {
            "enabled": bool(args.warm_graph),
            "cells": graph_warmup_cells,
        },
        "fixed_d3_exclusions": fixed_d3_exclusions,
        "memory_caps": memory_caps,
        "memory": {"after_load": after_load_memory, "after_run": after_run_memory},
        "draft_lm_head": draft_head,
        "candidate_environment": candidate_environment,
        "extra_environment": extra_environment,
        # The raw --env passthrough, every key of it. Without this a candidate
        # armed through --candidate-extra-env is invisible in the receipt
        # (2026-09-01); the allowlisted MTPLX_* keys additionally carry
        # requested-vs-effective and the file that reads them.
        "process_environment_overrides": process_environment_overrides,
        "paired_routed_glu": {
            "expected": candidate_environment.get("MTPLX_QWEN4_M4_ROUTED_GLU")
            == "1",
            "observed": bool(
                (getattr(runtime, "qwen4_m4_stage3_report", None) or {}).get(
                    "paired_routed_glu"
                )
            ),
        },
        "ple_prefix_reuse": {
            "expected": ple_prefix_reuse_expected,
            "observed": ple_prefix_reuse_observed,
        },
        "ple_m23_direct": {
            "expected": ple_m23_direct_expected,
            "observed": ple_m23_direct_observed,
        },
        "gdn_prefix_states": {
            "expected": gdn_prefix_states_expected,
            "observed": gdn_prefix_states_observed,
        },
        "guard": guard,
        "rows": rows,
    }
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.label
    )
    if args.receipt_path is not None:
        out = args.receipt_path
    else:
        if args.benchmark_matrix:
            suffix = "benchmark-matrix.json"
        elif args.natural_stop:
            suffix = f"variable-16k-max{args.max_tokens}.json"
        elif args.max_tokens != 1024:
            suffix = f"fixed-16k-output{args.max_tokens}.json"
        else:
            suffix = "seeds-16k-1k.json"
        out = OUT_DIR / f"abba-{args.sequence}-{safe_label}-{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[fable-abba] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
