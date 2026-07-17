#!/usr/bin/env python3
"""Shard a monolithic expert sidecar into Hub-uploadable record-aligned files.

``experts.bin`` sidecars (89.5-226 GB locally) exceed Hugging Face per-file
limits (50 GB hard cap, ~20 GB recommended), so this tool cuts the sidecar
at expert-record boundaries into ``experts-00001-of-000NN.bin`` files and
writes an updated manifest whose records carry per-record ``sidecar_shard``
names with shard-relative offsets.  The mtplx runtime reads the sharded
layout directly (``PositionalExpertReader``) — no reassembly after download.

Default is a DRY RUN that prints the shard plan; pass ``--execute`` to copy
bytes and publish ``expert-manifest-sharded.json``.  Executing needs enough
free disk for a full second copy of the sidecar and hours of sustained SSD
reads for the real artifacts: run it only in a guarded window.

Example (plan only):

    python scripts/shard_expert_sidecar.py \
        ~/.cache/huggingface/hy3-expert-only-mlx-q2 --max-shard-gb 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    DEFAULT_MAX_SIDECAR_SHARD_BYTES,
    ExpertManifestError,
    load_expert_manifest,
    plan_sidecar_shards,
    save_expert_manifest,
    write_sidecar_shards,
)


def _format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    return f"{value} B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        type=Path,
        help="model artifact root holding the manifest and sidecar",
    )
    parser.add_argument(
        "--manifest",
        default="expert-manifest.json",
        help="manifest file name inside the root (default: expert-manifest.json)",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--max-shard-bytes",
        type=int,
        default=None,
        help="maximum shard size in bytes (records are never split)",
    )
    size.add_argument(
        "--max-shard-gb",
        type=float,
        default=None,
        help="maximum shard size in GiB (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for shard files (default: the artifact root)",
    )
    parser.add_argument(
        "--output-manifest",
        default="expert-manifest-sharded.json",
        help="sharded manifest name (default: expert-manifest-sharded.json)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="copy shard bytes and write the sharded manifest (default: dry run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest_path = root / args.manifest
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        manifest = load_expert_manifest(manifest_path)
    except ExpertManifestError as exc:
        print(f"error: could not load manifest: {exc}", file=sys.stderr)
        return 2
    if manifest.sidecar is None:
        print("error: manifest has no single-file sidecar to shard", file=sys.stderr)
        return 2
    if args.max_shard_bytes is not None:
        max_shard_bytes = args.max_shard_bytes
    elif args.max_shard_gb is not None:
        max_shard_bytes = int(args.max_shard_gb * 1024**3)
    else:
        max_shard_bytes = DEFAULT_MAX_SIDECAR_SHARD_BYTES

    try:
        plans = plan_sidecar_shards(manifest, max_shard_bytes=max_shard_bytes)
    except ExpertManifestError as exc:
        print(f"error: could not plan shards: {exc}", file=sys.stderr)
        return 2

    mode = "EXECUTE" if args.execute else "DRY RUN"
    total = sum(plan.length for plan in plans)
    print(f"[{mode}] shard plan for {manifest.sidecar.file} in {root}")
    print(
        f"  sidecar: {_format_bytes(manifest.sidecar.size)}, "
        f"{len(manifest.records)} records, "
        f"alignment {manifest.sidecar.alignment}"
    )
    print(
        f"  plan: {len(plans)} shard(s) <= {_format_bytes(max_shard_bytes)} "
        f"each, {_format_bytes(total)} total"
    )
    for plan in plans:
        print(
            f"    {plan.name}  {_format_bytes(plan.length)}  "
            f"records {plan.record_indices[0]}..{plan.record_indices[-1]}  "
            f"source [{plan.source_offset}, {plan.source_offset + plan.length})"
        )
    if not args.execute:
        print("  dry run: nothing written; pass --execute to copy shard bytes")
        print(
            "  NOTE: executing re-copies the whole sidecar (needs equal free "
            "disk) and sustains heavy SSD reads; run in a guarded window."
        )
        return 0

    started = time.perf_counter()
    try:
        sharded = write_sidecar_shards(
            manifest,
            root,
            plans,
            output_dir=args.output_dir,
        )
    except ExpertManifestError as exc:
        print(f"error: shard write failed: {exc}", file=sys.stderr)
        return 1
    output_root = (
        root if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    output_manifest = output_root / args.output_manifest
    save_expert_manifest(sharded, output_manifest)
    elapsed = time.perf_counter() - started
    rate = total / 1024**2 / elapsed if elapsed else 0.0
    print(
        f"  wrote {len(plans)} shard(s) and {output_manifest.name} "
        f"in {elapsed:.1f}s ({rate:.0f} MiB/s)"
    )
    print(
        "  the original sidecar was left untouched; verify the sharded "
        "layout before reclaiming it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
