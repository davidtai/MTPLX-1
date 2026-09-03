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

``overlap@N`` Flag on at prefix depth N (``--layers``, default ``1,2,3,4``).
             The window is two compiled calls: an N-layer prefix, queued the
             statement after ``verify_input_array`` so the GPU has ~0.53 ms
             per prefix layer to run under the host's snapshot, PLE row read
             and suffix replay; then the (48-N)-layer suffix.
             ``prefix_build_ms`` + ``suffix_build_ms`` per window is the same
             host work, split.

             At N = 1 the prefix reads no PLE auxiliary (the single PLE layer
             is index 1) and the auxiliary is built in the join, exactly where
             the shipped route builds it.  At N > 1 the prefix CONTAINS the
             PLE layer, so the auxiliary is hoisted to the enqueue -- built
             once per window either way; ``aux_hoisted`` in the receipt should
             equal the window count on those arms and be 0 at N = 1.

             The predicted ceiling is ``min(N x 0.53, (48-N)/48 x 1.93)``
             ms/cycle -- 0.53 / 1.06 / 1.59 / 1.77 at N = 1/2/3/4 -- so a
             sweep that does not rise from N=1 to N=3 is telling you the
             per-layer GPU estimate (verify-body GPU / 48) is wrong, and one
             that rises and then falls at N=4 is telling you the host build
             left to hide under has run out.

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
           --prompt-tokens 16384 --max-tokens 192 --reps 2 --layers 1,2,3,4 \\
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


def _set_lane(enabled: bool, layer_count: int = 1) -> None:
    """Arm or disarm the lane, at one prefix depth, for the NEXT generate.

    ``arm_fixed_m4_graph_build_overlap`` forces a recompile of the pair at
    every request setup, so changing ``layer_count`` between arms really does
    change the graph the next arm replays -- it does not reuse the previous
    arm's partition.
    """

    from mtplx import graph_build_overlap as lane

    os.environ["MTPLX_FABLE_GRAPH_BUILD_OVERLAP"] = "1" if enabled else "0"
    os.environ["MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS"] = str(int(layer_count))
    lane.enabled.cache_clear()
    lane.items.cache_clear()
    lane.timing_enabled.cache_clear()
    lane.layers.cache_clear()
    lane.reset_receipt()


def _run_arm(
    *,
    runtime: Any,
    cell: dict[str, Any],
    seed: int,
    enabled: bool,
    driver_args: Any,
    layer_count: int = 1,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx import graph_build_overlap as lane
    from mtplx import ple_boundary
    from mtplx.generation import generate_mtpk
    from scripts.fable import abba_driver

    _set_lane(enabled, layer_count)
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
        "arm": f"overlap@{int(layer_count)}" if enabled else "stock",
        "armed": bool(enabled),
        "layers": int(layer_count) if enabled else 0,
        # The depth the bank actually INSTALLED.  A row whose `arm` says 3 and
        # whose `installed_layers` says 1 measured a different partition than
        # its label claims and its delta belongs to no arm.
        "installed_layers": int(overlap["prefix_layers"]),
        "aux_hoisted": int(overlap["aux_hoisted"]),
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


#: ``min(N x 0.53, (48-N)/48 x 1.93)`` -- the W63 census's per-layer GPU
#: estimate (verify-body GPU / 48) against the host build it can hide under.
#: Printed beside the measurement so an arm that lands nowhere near its own
#: prediction is visible without arithmetic.
def _predicted_saving_ms(layer_count: int) -> float:
    return min(layer_count * 0.53, (48 - layer_count) / 48 * 1.934)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _median(arm: str, field: str) -> float | None:
        values = [
            row[field]
            for row in rows
            if row["arm"] == arm and row.get(field) is not None
        ]
        return statistics.median(values) if values else None

    arms = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])
    overlap_arms = [arm for arm in arms if arm != "stock"]

    summary: dict[str, Any] = {"arms": arms}
    for field in (
        "ms_per_window",
        "decode_tok_s",
        "host_build_ms",
        "monolithic_build_ms",
        "prefix_build_ms",
        "suffix_build_ms",
    ):
        stock = _median("stock", field)
        entry: dict[str, Any] = {"stock": stock}
        for arm in overlap_arms:
            value = _median(arm, field)
            entry[arm] = value
            entry[f"{arm}_delta"] = (
                None if stock is None or value is None else value - stock
            )
        summary[field] = entry

    # The verdict table: one row per depth, measured against its own ceiling.
    by_depth = []
    stock_ms = _median("stock", "ms_per_window")
    for arm in overlap_arms:
        depth = int(arm.split("@")[1]) if "@" in arm else 1
        measured = _median(arm, "ms_per_window")
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_depth.append(
            {
                "arm": arm,
                "layers": depth,
                "installed_layers": sorted(
                    {row["installed_layers"] for row in arm_rows}
                ),
                "predicted_saving_ms": round(_predicted_saving_ms(depth), 3),
                "measured_saving_ms": None
                if stock_ms is None or measured is None
                else stock_ms - measured,
                "monolithic_windows": sum(
                    int(row["graph_build_overlap"]["monolithic_windows"])
                    for row in arm_rows
                ),
                "prefix_discarded": sum(
                    int(row["graph_build_overlap"]["prefix_discarded"])
                    for row in arm_rows
                ),
                "aux_hoisted": sum(int(row["aux_hoisted"]) for row in arm_rows),
                "windows": sum(int(row["windows"]) for row in arm_rows),
                "token_digest_matches_stock": all(
                    row["response_token_sha256"]
                    == next(
                        r["response_token_sha256"]
                        for r in rows
                        if r["arm"] == "stock"
                    )
                    for row in arm_rows
                )
                if any(row["arm"] == "stock" for row in rows)
                else None,
            }
        )
    summary["by_depth"] = by_depth

    digests: dict[str, set] = {}
    for row in rows:
        digests.setdefault(row["arm"], set()).add(row["response_token_sha256"])
    summary["token_digests"] = {
        arm: sorted(values) for arm, values in digests.items()
    }
    stock_digests = digests.get("stock")
    summary["token_digest_match"] = (
        None
        if not stock_digests or len(overlap_arms) == 0
        else all(digests[arm] == stock_digests for arm in overlap_arms)
    )
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
    parser.add_argument(
        "--layers",
        default="1,2,3,4",
        help=(
            "Prefix depths to price, comma separated "
            "(MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS).  Each depth gets its "
            "own overlap arm, ABBA-ordered against one shared stock arm per "
            "rep, so one guarded window prices the whole sweep."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--allow-unguarded", action="store_true")
    args = parser.parse_args(argv)

    depths = tuple(
        int(part.strip()) for part in str(args.layers).split(",") if part.strip()
    )
    if not depths or any(depth < 1 for depth in depths):
        parser.error(f"--layers must be one or more positive integers, got {args.layers!r}")

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

    # One stock arm plus one overlap arm per depth, per rep.  The order is
    # reversed on odd reps so no depth sits permanently on the warm or the
    # cold side of the stock arm (the same reason the two-arm version was
    # ABBA-ordered).
    plan: list[tuple[bool, int]] = [(False, 0)] + [(True, d) for d in depths]

    rows: list[dict[str, Any]] = []
    for rep in range(int(args.reps)):
        order = plan if rep % 2 == 0 else list(reversed(plan))
        for enabled, depth in order:
            row = _run_arm(
                runtime=runtime,
                cell=cell,
                seed=args.seed,
                enabled=enabled,
                driver_args=driver_args,
                layer_count=max(1, depth),
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
                            "installed_layers",
                            "aux_hoisted",
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
        "layers": list(depths),
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
        diverged = [
            entry["arm"]
            for entry in summary["by_depth"]
            if entry["token_digest_matches_stock"] is False
        ]
        print(
            "[micro_graph_build_overlap] WARNING: these arms produced "
            f"different tokens than stock: {diverged} -- the compile seam is "
            "NOT bit-neutral at those depths and no timing number above is a "
            "lever for them",
            file=sys.stderr,
            flush=True,
        )
    for entry in summary["by_depth"]:
        if entry["installed_layers"] not in ([entry["layers"]], []):
            print(
                "[micro_graph_build_overlap] WARNING: "
                f"{entry['arm']} installed depth(s) {entry['installed_layers']} "
                "-- this row measured a partition its label does not name",
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
