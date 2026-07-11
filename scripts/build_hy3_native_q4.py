#!/usr/bin/env python3
"""Build a compact, fully provenanced native Q4 artifact from Tencent Hy3.

The output contains resident-only safetensors files, a 16-KiB-aligned
``experts.bin`` for layers 1..80, and the bit-exact BF16 layer-80 head.  It
never creates a full Q4 checkpoint or ``layer80-q4.safetensors``.

The hidden sibling work directory and append-only checkpoint make the job
safe to resume by rerunning the same command.  The final directory appears
only after an atomic rename.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.hy3_native_quantizer import (  # noqa: E402
    DEFAULT_ALIGNMENT,
    ORACLE_REVISION,
    SOURCE_REVISION,
    Hy3NativeConversionError,
    build_conversion_plan,
    execute_conversion,
    plan_summary,
    prepare_conversion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Official tencent/Hy3 BF16 local directory",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        required=True,
        help="Pinned pipenetwork/Hy3-4bit layout-oracle snapshot",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--oracle-revision", default=ORACLE_REVISION)
    parser.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT)
    parser.add_argument(
        "--bf16-head",
        type=Path,
        help="Verified consolidated layer80-bf16.safetensors (defaults under --source)",
    )
    parser.add_argument(
        "--copy-bf16-head",
        action="store_true",
        help=(
            "Explicitly allocate a copy of the ~7.5-GB BF16 head. The default "
            "requires a same-volume hardlink."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Inspect all headers and print the plan without hashing or MLX compute",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Hash all source shards and bind/verify the BF16 head, then stop "
            "before the GPU run-lock or MLX import"
        ),
    )
    return parser


def _required_output_bytes(plan, *, copy_bf16_head: bool, bf16_head: Path) -> int:
    resident_payload = sum(
        tensor.nbytes for output in plan.resident_files for tensor in output.tensors
    )
    head_bytes = (
        bf16_head.stat().st_size if copy_bf16_head and bf16_head.is_file() else 0
    )
    # Account for safetensors headers/manifests/checkpoint and leave a small
    # operational cushion without pretending to estimate the source download.
    return plan.experts_file_size + resident_payload + head_bytes + 2 * 1024**3


def main() -> int:
    args = build_parser().parse_args()
    try:
        plan = build_conversion_plan(
            args.source,
            args.oracle,
            source_revision=args.source_revision,
            oracle_revision=args.oracle_revision,
            alignment=args.alignment,
        )
        summary = plan_summary(plan)
        head = (
            args.bf16_head.expanduser().resolve()
            if args.bf16_head is not None
            else plan.source_root / "layer80-bf16.safetensors"
        )
        required = _required_output_bytes(
            plan,
            copy_bf16_head=args.copy_bf16_head,
            bf16_head=head,
        )
        output_parent = args.output.expanduser().resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(output_parent).free
        summary["minimum_free_output_bytes"] = required
        summary["available_output_bytes"] = free
        summary["bf16_head_storage"] = "copy" if args.copy_bf16_head else "hardlink"
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.plan_only:
            return 0
        if free < required:
            raise Hy3NativeConversionError(
                f"output filesystem has {free:,} free bytes; compact conversion "
                f"requires at least {required:,}"
            )
        if args.prepare_only:
            work = prepare_conversion(
                plan,
                args.output,
                bf16_head_source=head,
                copy_bf16_head=args.copy_bf16_head,
                on_event=lambda event: print(
                    json.dumps(event, sort_keys=True),
                    flush=True,
                ),
            )
            print(
                json.dumps(
                    {
                        "prepared_work_directory": str(work),
                        "gpu_compute_started": False,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0
        result = execute_conversion(
            plan,
            args.output,
            bf16_head_source=head,
            copy_bf16_head=args.copy_bf16_head,
            on_event=lambda event: print(
                json.dumps(event, sort_keys=True),
                flush=True,
            ),
        )
    except Hy3NativeConversionError as exc:
        print(f"Hy3 native conversion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(result)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
