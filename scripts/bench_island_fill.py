#!/usr/bin/env python3
"""Time DenseIslandStore.fill across worker/coalesce configurations.

Measures the island fill path only (grouped sidecar reads + SHA-256 +
component-bank writes) over a real artifact, sweeping:

  * ``--workers``      parallel per-layer fill threads (1 = the old serial path)
  * ``--coalesce-mb``  0 = scatter-preadv fill; N = sequential bounce-buffer
                       reads sliced into the banks (native backend eligible)

DO NOT run this while anything else owns the machine: it allocates the full
island bank set in MLX memory (layers x experts x record bytes) and sustains
SSD reads at device bandwidth.  Run it in a guarded window only.

Cold-cache guidance: successive configurations re-read the same file ranges,
so the macOS page cache warms them.  ``--bypass-page-cache`` sets F_NOCACHE
on the reader descriptors for a fair repeatable comparison without sudo; for
true cold numbers, alternate with ``sudo purge``.

Example (hy3 Q2, first 8 island layers, both fill modes):

    python scripts/bench_island_fill.py \
        ~/.cache/huggingface/hy3-expert-only-mlx-q2 \
        --layer-count 8 --workers 1,2,4,6,8 --coalesce-mb 0,32 --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtplx.expert_io import PositionalExpertReader  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="model artifact root")
    parser.add_argument(
        "--manifest",
        default="expert-manifest.json",
        help="manifest name inside the root (default: expert-manifest.json)",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=4,
        help="number of manifest layers to fill, lowest first (default: 4)",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="explicit comma-separated layer ids (overrides --layer-count)",
    )
    parser.add_argument(
        "--workers",
        type=_parse_int_list,
        default=(1, 2, 4, 6, 8),
        help="worker counts to sweep (default: 1,2,4,6,8)",
    )
    parser.add_argument(
        "--coalesce-mb",
        type=_parse_int_list,
        default=(0, 32),
        help="coalesce chunk MiB values to sweep; 0 = scatter (default: 0,32)",
    )
    parser.add_argument(
        "--no-verify-hash",
        action="store_true",
        help="skip SHA-256 verification (isolates raw IO time)",
    )
    parser.add_argument(
        "--bypass-page-cache",
        action="store_true",
        help="set F_NOCACHE on reader descriptors (repeatable cold-ish reads)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repetitions per configuration (default: 1)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write machine-readable results to this file",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm that the machine is free (guarded window)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest = load_expert_manifest(root / args.manifest)
    layer_ids = sorted({record.layer for record in manifest.records})
    if args.layers:
        layers = tuple(int(item) for item in args.layers.split(","))
        missing = sorted(set(layers) - set(layer_ids))
        if missing:
            print(f"error: layers {missing} are not in the manifest", file=sys.stderr)
            return 2
    else:
        layers = tuple(layer_ids[: max(1, args.layer_count)])
    expert_count = sum(1 for record in manifest.records if record.layer == layers[0])
    record_bytes = manifest.records[0].logical_bytes
    island_bytes = len(layers) * expert_count * record_bytes
    print(
        f"island fill benchmark: {len(layers)} layer(s) x {expert_count} experts "
        f"x {record_bytes / 1024**2:.1f} MiB = {island_bytes / 1024**3:.2f} GiB "
        f"of MLX bank allocation per configuration"
    )
    if not args.yes:
        print(
            "refusing to run without --yes: this allocates the bank bytes above "
            "and reads the SSD at device bandwidth. Confirm the machine is in a "
            "guarded window (no other benchmark, server, or model load).",
            file=sys.stderr,
        )
        return 2

    from mtplx.models.expert_mlx import DenseIslandStore  # imports MLX

    verify_hash = not args.no_verify_hash
    results = []
    for coalesce_mb in args.coalesce_mb:
        coalesce = coalesce_mb * 1024 * 1024 if coalesce_mb > 0 else None
        for workers in args.workers:
            for repeat in range(max(1, args.repeats)):
                store = DenseIslandStore(
                    manifest,
                    layers,
                    expert_count=expert_count,
                )
                reader = PositionalExpertReader(
                    root,
                    bypass_page_cache=args.bypass_page_cache,
                )
                try:
                    started = time.perf_counter()
                    store.fill(
                        manifest,
                        reader,
                        verify_hash=verify_hash,
                        max_workers=workers,
                        coalesce_chunk_bytes=coalesce,
                    )
                    elapsed = time.perf_counter() - started
                finally:
                    metrics = reader.metrics.as_dict()
                    store.close()
                    reader.close()
                throughput = island_bytes / 1024**3 / elapsed if elapsed else 0.0
                row = {
                    "workers": workers,
                    "coalesce_mb": coalesce_mb,
                    "verify_hash": verify_hash,
                    "bypass_page_cache": args.bypass_page_cache,
                    "repeat": repeat,
                    "seconds": round(elapsed, 3),
                    "gib_per_s": round(throughput, 3),
                    "preadv_calls": metrics["python_preadv_invocations"],
                    "native_calls": metrics["native_positional_calls"],
                    "read_bytes": metrics["read_bytes"],
                }
                results.append(row)
                print(
                    f"  workers={workers:2d} coalesce={coalesce_mb:3d}MiB "
                    f"repeat={repeat}  {elapsed:7.2f}s  {throughput:6.2f} GiB/s  "
                    f"(preadv={row['preadv_calls']}, native={row['native_calls']})"
                )
    if args.json:
        payload = {
            "root": str(root),
            "layers": list(layers),
            "expert_count": expert_count,
            "record_bytes": record_bytes,
            "island_bytes": island_bytes,
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
