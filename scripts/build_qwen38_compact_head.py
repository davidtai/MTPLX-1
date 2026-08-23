#!/usr/bin/env python3
"""Build the proposal-only Q2 table for the pinned local Qwen 3.8 model."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path.home() / (
    ".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
)
DEFAULT_OUTPUT = (
    Path.home()
    / ".cache/mtplx/qwen38-optimized-speed-compact-q2-v1/model.safetensors"
)
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing artifact after re-validating the model contract",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    from mtplx.artifacts import load_config
    from mtplx.qwen38_challenge import validate_qwen38_27b_contract
    from mtplx.qwen38_compact_head import (
        build_qwen38_compact_artifact,
        validate_qwen38_compact_artifact,
    )

    config = load_config(model_path)
    contract = validate_qwen38_27b_contract(config, model_path)
    if output_path.exists() and not args.force:
        artifact = validate_qwen38_compact_artifact(
            output_path,
            source_contract_id=contract.contract_id,
        )
    else:
        lock_handle = args.lock.open("a+")
        try:
            fcntl.lockf(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {args.lock}") from exc

        from mtplx.runtime import load

        runtime = load(model_path, mtp=True)
        text = getattr(runtime.model, "language_model", runtime.model)
        artifact = build_qwen38_compact_artifact(
            text.lm_head,
            output_path,
            source_contract_id=contract.contract_id,
        )

    print(
        json.dumps(
            {
                "bytes": artifact.bytes,
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "source_contract_id": artifact.source_contract_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
