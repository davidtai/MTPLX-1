#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script loads the production model and
# runs two real decodes; running it outside the serialized window interrupts
# whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price W63's graph-build overlap on the real fixed-M4 verify graph.

WHAT IT MEASURES

The retained-stack control census (w58-retained-control-census-1788370322,
382 cycles) puts 1.934 ms/cycle of GPU idle -- 86.9 % host-late, in 382 of 382
cycles -- immediately before the compiled verify body (a fixed 3,668
dispatches before the cycle's lm_head).  That is 26.5 % of the stack's
7.301 ms/cycle of idle, and it is one thing: the host-side replay of the
``mx.compile``d physical-M4 verify graph -- ~5,200 nodes and ~137 array inputs
walked on the generation thread while the GPU holds nothing.  That number
cannot be reproduced with synthetic tensors at "production shapes"; it is a
property of the real graph, so this bench loads the real model.

Two arms, in ONE process, on the same loaded runtime and the same prompt:

``stock``    ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP`` off.  The window is one
             compiled call.  ``graph_build_ms / graph_build_calls`` (the W62
             ``timing`` instrument, armed here on both arms) is that call's
             host seconds -- the census's gap-B host term, measured from
             inside the process.

``overlap``  Flag on.  The window is two compiled calls: a ~110-node layer-0
             prefix, queued the statement after ``verify_input_array`` so the
             GPU has ~0.53 ms of work to run under the host's snapshot, PLE
             row read and suffix replay; then the ~5,090-node suffix.
             ``prefix_build_ms`` + ``suffix_build_ms`` per window is the same
             host work, split.

The host columns answer "did the split move the host cost, or only the GPU
work?" -- the honest expectation is that the host total is UNCHANGED or
slightly worse (two tree-flattens, two ``async_eval``s), and that the whole
win is GPU work moved under host time already being spent.  ``ms_per_window``
is the number that decides the lever; the ABBA is still the verdict.

WHAT IT DOES NOT MEASURE

Bit-identity.  The seam puts an ``mx.compile`` boundary between layer 0 and
layer 1, and MLX can fuse an element-wise chain across it in the monolithic
graph.  Only ``scripts/fable/abba_window.py``'s ``response_token_sha256`` and
per-cycle ``accepted`` list settle that; a delta here on a diverged arm is
worthless.  This bench prints the two arms' token digests so a divergence is
at least VISIBLE before the window is spent, but n=1 short decodes at
temperature 1.0 with the same seed are the weak form of that check.

RUN IT (guarded)::

    W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w63-graph-build-overlap
    PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
    RG=/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py
    PLIST=/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist
    mkdir -p $W/.benchmark-artifacts/fable
    env PYTHONPATH=$W $PY $RG --plist $PLIST --lock-timeout-seconds 3600 \\
        --child-timeout-seconds 3600 \\
      -- env PYTHONPATH=$W $PY $W/scripts/fable/micro_graph_build_overlap.py \\
           --prompt-tokens 16384 --max-tokens 192 --reps 2 \\
           --json $W/.benchmark-artifacts/fable/micro-graph-build-overlap.json

Read ``windows`` first: an arm whose ``overlap.suffix_build_calls`` is not
equal to its window count ran the shipped monolithic route for the difference
and its delta is diluted by exactly that fraction.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
GUARD_ATTEST_FD_ENV = "MTPLX_GUARD_ATTEST_FD"
BANNER = (
    "[micro_graph_build_overlap] GPU WINDOW REQUIRED -- this loads the "
    f"production model and decodes; run under {LOCK_PATH} via "
    "bench/laguna/run_guarded.py (or pass --allow-unguarded)"
)

#: Exactly the ABBA control's driver flags (``abba_window.CONTROL_FLAGS``
#: minus ``--max-tokens``, which this bench owns), so the graph this bench
#: replays is the graph the A/B replays.  A drift here makes every number
#: below a number about a different graph.
CONTROL_DRIVER_ARGV = (
    "--target-mode", "batched",
    "--require-compiled-verify",
    "--m4-stage3",
    "--qsa-fused-kv-gather",
    "--full-frspec",
    "--compiled-mtp-prepare",
    "--prewarm-ngram-table",
)


def _apply_driver_environment(source_path: Path, argv: tuple[str, ...]) -> Any:
    """Reproduce ``abba_driver.main``'s construction environment exactly.

    Imported from the driver rather than restated: the profile keys, the
    turbo baseline and the fixed-D3 exclusions are the driver's contract, and
    a bench that guesses them measures a different runtime than the A/B.
    """

    from scripts.fable import abba_driver

    args = abba_driver.build_parser().parse_args(
        [
            *argv,
            "--label", "micro-graph-build-overlap",
            "--sequence", "0",
            "--seed", "20260829",
            "--source", str(source_path),
            "--max-tokens", "1",
        ]
    )
    family_overrides, _candidate_environment = abba_driver.build_family_overrides(
        args
    )
    from mtplx.profiles import apply_profile_env, get_profile

    expected_environment = get_profile("turbo").env_dict()
    expected_environment.update(family_overrides)
    for key in expected_environment:
        os.environ.pop(key, None)
    apply_profile_env("turbo", runtime_env_overrides=family_overrides)
    os.environ.pop("MTPLX_QWEN4_M4_DOWN_FUSED", None)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "MTPLX_CONTEXT_WINDOW_TOKENS": "262144",
            "MTPLX_NGRAM_HOT_MB": "1024",
            "MTPLX_MEMORY_LIMIT_BYTES": str(abba_driver.MEMORY_LIMIT_BYTES),
            "MTPLX_WIRED_LIMIT_BYTES": str(abba_driver.WIRED_LIMIT_BYTES),
            "MTPLX_ADAPTIVE_DTEMP": "0",
            "MTPLX_STATE_REBASE_EVERY": "0",
            "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "0",
            # --full-frspec, as the control arm carries it.
            "MTPLX_FRSPEC_DRAFT": "1",
            "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
        }
    )
    observed = {key: os.environ.get(key) for key in expected_environment}
    drift = {
        key: (expected_environment[key], observed[key])
        for key in expected_environment
        if observed[key] != expected_environment[key]
    }
    if drift:
        raise RuntimeError(f"construction environment drifted: {drift}")
    return args


def _arm_host_timers() -> None:
    """Turn on BOTH host build timers, for both arms, for the whole process.

    ``mtplx.graphbank`` reads them as module constants at import, so a bench
    that wants them on every arm sets them here rather than through the two
    lanes' item flags -- which would otherwise put the instrument on one arm
    and not the other.  Cost is two ``perf_counter`` pairs per window.
    """

    from mtplx import graphbank

    graphbank._PLE_BOUNDARY_GRAPH_TIMING = True
    graphbank._GRAPH_BUILD_OVERLAP_TIMING = True


def _set_lane(enabled: bool) -> None:
    """Arm or disarm the W63 lane for the NEXT ``generate_mtpk`` call."""

    from mtplx import graph_build_overlap as lane

    os.environ["MTPLX_FABLE_GRAPH_BUILD_OVERLAP"] = "1" if enabled else "0"
    lane.enabled.cache_clear()
    lane.items.cache_clear()
    lane.timing_enabled.cache_clear()
    lane.reset_receipt()


def _run_arm(
    *,
    runtime: Any,
    cell: dict[str, Any],
    seed: int,
    enabled: bool,
    driver_args: Any,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx import graph_build_overlap as lane
    from mtplx import ple_boundary
    from mtplx.generation import generate_mtpk
    from scripts.fable import abba_driver

    _set_lane(enabled)
    ple_boundary.reset_receipt()
    abba_driver.reset_run_caches(runtime, mx)
    mx.reset_peak_memory()

    started = time.perf_counter()
    output = generate_mtpk(
        runtime,
        cell["prompt_ids"],
        max_tokens=int(cell["max_tokens"]),
        sampler=cell["sampler"],
        draft_sampler=cell["sampler"],
        speculative_depth=3,
        seed=seed,
        stop_token_ids=set(),
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy=driver_args.verify_strategy,
        verify_core="linear-gdn-from-conv-tape",
        draft_core=driver_args.draft_core,
    )
    wall_s = time.perf_counter() - started

    stats = output.stats
    overlap = lane.last_receipt()
    boundary = ple_boundary.last_receipt()
    windows = int(overlap["suffix_joined"]) + int(overlap["monolithic_windows"])
    if not windows:
        windows = int(boundary["graph_build_calls"])

    def _per_window(value: float) -> float | None:
        return None if not windows else float(value) / windows

    import hashlib

    return {
        "arm": "overlap" if enabled else "stock",
        "armed": bool(enabled),
        "wall_s": wall_s,
        "decode_elapsed_s": float(stats.decode_elapsed_s),
        "decode_tok_s": float(stats.decode_tok_s),
        "generated_tokens": int(len(output.tokens)),
        "windows": windows,
        # The host term the census calls gap B, per window.
        "monolithic_build_ms": _per_window(boundary["graph_build_ms"]),
        "monolithic_build_calls": int(boundary["graph_build_calls"]),
        "prefix_build_ms": _per_window(overlap["prefix_build_ms"]),
        "suffix_build_ms": _per_window(overlap["suffix_build_ms"]),
        "host_build_ms": _per_window(
            float(boundary["graph_build_ms"])
            + float(overlap["prefix_build_ms"])
            + float(overlap["suffix_build_ms"])
        ),
        "ms_per_window": None
        if not windows
        else float(stats.decode_elapsed_s) * 1000.0 / windows,
        "graph_build_overlap": overlap,
        "response_token_sha256": hashlib.sha256(
            ",".join(str(int(token)) for token in output.tokens).encode()
        ).hexdigest(),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _median(arm: str, field: str) -> float | None:
        values = [
            row[field]
            for row in rows
            if row["arm"] == arm and row.get(field) is not None
        ]
        return statistics.median(values) if values else None

    summary: dict[str, Any] = {}
    for field in (
        "ms_per_window",
        "decode_tok_s",
        "host_build_ms",
        "monolithic_build_ms",
        "prefix_build_ms",
        "suffix_build_ms",
    ):
        stock = _median("stock", field)
        overlap = _median("overlap", field)
        summary[field] = {
            "stock": stock,
            "overlap": overlap,
            "delta": None
            if stock is None or overlap is None
            else overlap - stock,
        }
    digests = {row["arm"]: row["response_token_sha256"] for row in rows}
    summary["token_digest_match"] = (
        digests.get("stock") == digests.get("overlap")
        if len(digests) == 2
        else None
    )
    summary["token_digests"] = digests
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="ABBA-ordered arm pairs (stock, overlap, overlap, stock, ...).",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--allow-unguarded", action="store_true")
    args = parser.parse_args(argv)

    if not args.allow_unguarded and not os.environ.get(GUARD_ATTEST_FD_ENV):
        print(BANNER, file=sys.stderr)
        return 2

    source_path = args.source.resolve(strict=True)
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    driver_args = _apply_driver_environment(source_path, CONTROL_DRIVER_ARGV)

    from mtplx.server.openai import _apply_metal_memory_caps
    from scripts.fable import abba_driver

    caps = _apply_metal_memory_caps(
        minimum_resident_bytes=abba_driver.MINIMUM_RESIDENT_BYTES
    )
    if not caps.get("applied"):
        raise RuntimeError(f"Metal memory caps did not apply: {caps}")

    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    runtime = load(driver_args.model.resolve(strict=True), mtp=True)
    prewarm = abba_driver.prewarm_ngram_table(
        driver_args.model.resolve(strict=True) / abba_driver.NGRAM_TABLE_NAME
    )
    print(
        "[micro_graph_build_overlap] n-gram prewarm "
        + json.dumps(prewarm, sort_keys=True),
        flush=True,
    )
    _arm_host_timers()

    cell = abba_driver.build_production_cell(
        runtime,
        SamplerConfig,
        label="micro-graph-build-overlap",
        max_tokens=int(args.max_tokens),
        prompt_tokens=int(args.prompt_tokens),
    )

    # One unmeasured warm-up so the compiled graphs, the first prefill chunk
    # and the n-gram pages are warm before either arm is timed.
    _run_arm(
        runtime=runtime,
        cell=cell,
        seed=args.seed,
        enabled=False,
        driver_args=driver_args,
    )

    rows: list[dict[str, Any]] = []
    for rep in range(int(args.reps)):
        order = (False, True) if rep % 2 == 0 else (True, False)
        for enabled in order:
            row = _run_arm(
                runtime=runtime,
                cell=cell,
                seed=args.seed,
                enabled=enabled,
                driver_args=driver_args,
            )
            row["rep"] = rep
            rows.append(row)
            print(
                "[micro_graph_build_overlap] "
                + json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "arm",
                            "windows",
                            "ms_per_window",
                            "decode_tok_s",
                            "monolithic_build_ms",
                            "prefix_build_ms",
                            "suffix_build_ms",
                            "host_build_ms",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = _summarize(rows)
    payload = {
        "prompt_tokens": int(args.prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "reps": int(args.reps),
        "seed": int(args.seed),
        "source": str(source_path),
        "rows": rows,
        "summary": summary,
    }
    print(
        "[micro_graph_build_overlap] summary "
        + json.dumps(summary, indent=2, sort_keys=True),
        flush=True,
    )
    if summary["token_digest_match"] is False:
        print(
            "[micro_graph_build_overlap] WARNING: the two arms produced "
            "different tokens -- the layer-0/layer-1 compile seam is NOT "
            "bit-neutral here and no timing number below is a lever",
            file=sys.stderr,
            flush=True,
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[micro_graph_build_overlap] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
