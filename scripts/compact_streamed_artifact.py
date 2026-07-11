#!/usr/bin/env python3
"""Create an atomic resident-shards + expert-sidecar MTPLX artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.compact_streamed_artifact import (  # noqa: E402
    DEFAULT_CHUNK_BYTES,
    DEFAULT_MAX_SHARD_BYTES,
    CompactArtifactError,
    compact_streamed_artifact,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-bind legacy source shards, repack resident tensors with bounded "
            "raw-byte I/O, and hard-link the manifest-bound expert sidecar. The "
            "source is never deleted."
        )
    )
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("experts_bin", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-sidecar-copy",
        action="store_true",
        help=(
            "Explicitly allow bounded copies when sidecar/auxiliary hard links fail; "
            "without this flag the build fails instead of duplicating large files."
        ),
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_SHARD_BYTES,
    )
    parser.add_argument(
        "--chunk-bytes",
        type=_positive_int,
        default=DEFAULT_CHUNK_BYTES,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = compact_streamed_artifact(
            args.model_root,
            args.manifest,
            args.experts_bin,
            args.output,
            allow_sidecar_copy=args.allow_sidecar_copy,
            max_shard_bytes=args.max_shard_bytes,
            chunk_bytes=args.chunk_bytes,
        )
    except (CompactArtifactError, OSError) as exc:
        raise SystemExit(f"compact artifact build failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
