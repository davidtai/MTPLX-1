#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 **smallcode-redo** window
(2026-07-22): the small-code arm of record run against the THREE
already-done envelopes in one guarded window -- `hy3-oq2e-rq4-88e` (islands
79, full residency), `hy3-oq2e-rq4-80` (islands 69, 10 streamed),
`hy3-oq2e-rq4-64-cachelru` (ZERO islands, cache-heavy LRU) -- contexts=256,
output-tokens=256, K1 (+free AR), bf16 KV, 3 reps each.

David's correction (live, 2026-07-22): the per-envelope "small naturalistic
prompt" arm that already ran at 88/80/64 (t3_88col_naturalistic_*,
t3_80col_naturalistic_*, t3_64col_naturalistic_*) was ALWAYS meant to be a
SMALL CODE prompt, not prose -- this is a coding LLM, prose is out of
distribution. Those prose arms are downgraded to "prose reference --
incidental, not of record". The arm of record here uses the SAME
release-valid realistic_programming_v1 builder every context cell already
uses -- plain `--contexts 256`, NO custom prompt_tail, NO library-call
mechanics/monkeypatching (unlike research/t3-*/run_naturalistic_*_rep.py,
which exists ONLY because hy3-oq2e has no --prompt-tail CLI flag for a
genuinely custom prose tail; the small-code arm needs no such workaround --
it is just a smaller `--contexts` value against the SAME builder every
other matrix cell uses).

Sub-1024-contexts sanity check (cited, not assumed): scripts/
benchmark_q2_mtp_depth_matrix.py's `--contexts` uses `_integer_csv`
(build_parser, ~line 300-312), whose only per-value bound is `parsed <= 0`
-> argparse.ArgumentTypeError (~line 287-288) -- no minimum-value floor,
positive integers only. `_run_depth_matrix_impl`'s own gate (~line
2778-2784) re-checks `context_values` are unique positive integers --
same bound, no floor either. The ONLY other structural gate that touches
`contexts` is ~line 2803-2806:
    if max(context_values) + output_tokens > int(options["max_live_kv_tokens"]):
        raise BenchmarkConfigurationError(...)
At contexts=256, output-tokens=256, that is 512 tokens against the CLI's
own `--max-live-kv-tokens` default of 4096 (~line 382) -- comfortably
inside, gate does not fire, no override needed. Confirmed live (this
script, CPU only) against the REAL per-preset admission gate below: all
three presets ADMIT at the shared default kv=4096, byte-for-byte the same
control point every sibling column's own "1024" arm already clears.

Sibling of research/t3-64col/preflight.py and research/t3-80col/preflight.py
(same gate machinery, generalized here to 3 presets x 1 arm each instead of
1 preset x 4 arms); same guarantee: no MLX buffer is ever allocated, no
weight byte is ever read (ExpertStreamingRuntime.open / apis.load() are
never reached; the gate stops at `config.memory_plan()` / `plan.fits_fixed`,
via `research/envelope_admission_sweep.py`'s `evaluate_admission()` --
imported directly, never re-implemented).

MEASURED (this script, 2026-07-22): all three presets ADMIT DIRECTLY at
their own declared limit, no KV-delta override needed:
  88e:          fixed=95.6294 GiB  limit=96.0000 GiB  margin=+0.3706 GiB
  80:           fixed=86.1372 GiB  limit=87.0000 GiB  margin=+0.0223 GiB
  64-cachelru:  fixed=20.6411 GiB  limit=71.0000 GiB  margin=+0.3667 GiB
                (49.99 GiB elastic cache untouched at this KV level)

WIRED-KNOB CAP (mandatory, every t3-* window's launch instructions): any
arm whose derived override would exceed 112 GiB (114688 MiB, the
sanctioned wired knob) is marked SKIP -- never exceed the knob, regardless
of what the harness's own ADMIT/REJECT gate says. Not exercised here (all
three ADMIT with no override), but the machinery is retained for parity
with every sibling preflight in this family.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-smallcode-redo/preflight.py \\
        --output evals/tier2/t3_smallcode_redo_admission_preflight.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
for _path in (REPO_ROOT, RESEARCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from envelope_admission_sweep import (  # noqa: E402
    CONTROL_KV_TOKENS,
    KV_BYTES_PER_TOKEN,
    evaluate_admission,
)

GIB = 1024**3
WIRED_KNOB_LIMIT_GIB = 112.0  # 114688 MiB, sanctioned ceiling; never exceeded
SIGNOFF_CEILING_GIB = 90.0  # David's rule: >~90 GiB projected peak here needs a fresh ask
CONTEXTS = 256
OUTPUT_TOKENS = 256

# (arm_name, preset, contexts, output_tokens, max_live_kv_tokens or None-for-default)
ARMS: tuple[tuple[str, str, int, int, int | None], ...] = (
    ("88e", "hy3-oq2e-rq4-88e", CONTEXTS, OUTPUT_TOKENS, None),
    ("80", "hy3-oq2e-rq4-80", CONTEXTS, OUTPUT_TOKENS, None),
    ("64-cachelru", "hy3-oq2e-rq4-64-cachelru", CONTEXTS, OUTPUT_TOKENS, None),
)


def evaluate_arm(
    name: str, preset: str, contexts: int, output_tokens: int, kv_override_tokens: int | None
) -> dict[str, Any]:
    kv_tokens = kv_override_tokens if kv_override_tokens is not None else CONTROL_KV_TOKENS
    base = evaluate_admission(preset, kv_tokens)
    cell: dict[str, Any] = {
        "arm": name,
        "preset": preset,
        "contexts": contexts,
        "output_tokens": output_tokens,
        "max_live_kv_tokens": kv_tokens,
        "max_live_kv_tokens_is_override": kv_override_tokens is not None,
        "base": base,
    }

    if base["status"] == "ADMIT":
        cell["final_memory_limit_bytes"] = base["memory_limit_bytes"]
        cell["final_status"] = "ADMIT"
        cell["knob_exceeded"] = False
        return cell

    declared_limit_bytes = base["memory_limit_bytes"]
    kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
    override_bytes = declared_limit_bytes + kv_delta_tokens * KV_BYTES_PER_TOKEN
    override_gib = override_bytes / GIB
    cell["kv_delta_tokens"] = kv_delta_tokens
    cell["derived_override_bytes"] = override_bytes
    cell["derived_override_gib"] = override_gib

    if override_gib > WIRED_KNOB_LIMIT_GIB:
        cell["final_status"] = "SKIP"
        cell["knob_exceeded"] = True
        cell["skip_reason"] = (
            f"derived override {override_gib:.4f} GiB exceeds the sanctioned "
            f"{WIRED_KNOB_LIMIT_GIB:.1f} GiB wired knob -- never exceed the knob"
        )
        return cell

    overridden = evaluate_admission(preset, kv_tokens, memory_limit_override_bytes=override_bytes)
    cell["overridden"] = overridden
    cell["final_memory_limit_bytes"] = override_bytes
    cell["final_status"] = overridden["status"]
    cell["knob_exceeded"] = False
    if overridden["status"] != "ADMIT":
        cell["final_status"] = "SKIP"
        cell["skip_reason"] = (
            f"still REJECTs at the derived override ({override_gib:.4f} GiB): "
            f"excess {overridden.get('excess_bytes')} bytes -- preset's own "
            "fixed footprint over-books its declared limit chain independent "
            "of the wired knob"
        )
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cells = [evaluate_arm(*arm) for arm in ARMS]

    for cell in cells:
        status = cell["final_status"]
        limit_gib = cell.get("final_memory_limit_bytes", cell["base"]["memory_limit_bytes"]) / GIB
        fixed_gib = cell["base"]["fixed_bytes"] / GIB
        print(
            f"[preflight] arm={cell['arm']:12} preset={cell['preset']:24} contexts={cell['contexts']:6} "
            f"kv={cell['max_live_kv_tokens']:6} override={cell['max_live_kv_tokens_is_override']!s:5} "
            f"-> {status:6} fixed={fixed_gib:8.4f} GiB limit={limit_gib:8.4f} GiB "
            f"knob_exceeded={cell['knob_exceeded']}",
            file=sys.stderr,
        )
        if status != "SKIP" and limit_gib > SIGNOFF_CEILING_GIB:
            print(
                f"[preflight] WARNING: arm={cell['arm']} limit {limit_gib:.4f} GiB exceeds the "
                f"{SIGNOFF_CEILING_GIB:.1f} GiB sub-85/90 sign-off band -- confirm with David "
                "before launching this arm.",
                file=sys.stderr,
            )

    runnable = [c for c in cells if c["final_status"] != "SKIP"]
    all_admit = all(c["final_status"] == "ADMIT" for c in runnable)
    any_skip = any(c["final_status"] == "SKIP" for c in cells)

    payload = {
        "schema": "mtplx-t3-smallcode-redo-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight for the small-code arm of record "
            "(contexts=256, output-tokens=256, K1+AR, bf16 KV) run against the "
            "three already-done envelopes: hy3-oq2e-rq4-88e, hy3-oq2e-rq4-80, "
            "hy3-oq2e-rq4-64-cachelru. Supersedes the prose-naturalistic arm "
            "(David correction 2026-07-22) -- prose receipts already collected "
            "at these three envelopes are downgraded to incidental reference. "
            "Arms whose derived KV-delta override would exceed the 112 GiB "
            "wired knob, or that still REJECT even at that override, are "
            "marked SKIP and MUST NOT be run."
        ),
        "contexts": CONTEXTS,
        "output_tokens": OUTPUT_TOKENS,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "wired_knob_limit_gib": WIRED_KNOB_LIMIT_GIB,
        "signoff_ceiling_gib": SIGNOFF_CEILING_GIB,
        "cells": cells,
        "all_runnable_admit": all_admit,
        "any_skip": any_skip,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)

    return 0 if all_admit else 1


if __name__ == "__main__":
    raise SystemExit(main())
