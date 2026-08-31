#!/usr/bin/env python3
"""Launch the reviewed PR #391 benchmark driver with diagnostic row capture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import textwrap
from typing import Sequence


REVIEWED_DRIVER = Path("/private/tmp/pr391_fixed_d3_abba.py")
REVIEWED_DRIVER_SHA256 = (
    "0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0"
)
REVIEWED_SOURCE = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-flash-next-210-restack"
)
REVIEWED_SEEDS = (20260829, 20260830, 20260831)
INSTALL_ANCHOR = "    after_load_memory = {"
FINALIZE_ANCHOR = "    cell_receipts = []"


class DriverSourceMismatch(RuntimeError):
    """The benchmark driver no longer matches the reviewed source."""


class DriverArgumentMismatch(RuntimeError):
    """The delegated driver arguments do not describe the reviewed workload."""


def load_reviewed_driver(path: Path = REVIEWED_DRIVER) -> str:
    source = path.read_text()
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != REVIEWED_DRIVER_SHA256:
        raise DriverSourceMismatch(
            "benchmark driver SHA-256 mismatch: expected "
            f"{REVIEWED_DRIVER_SHA256}, got {actual}"
        )
    return source


def _insert_once(source: str, anchor: str, insertion: str) -> str:
    if source.count(anchor) != 1:
        raise DriverSourceMismatch(
            f"reviewed driver anchor count changed for {anchor!r}"
        )
    return source.replace(anchor, insertion + anchor, 1)


def transform_capture_driver(source: str, *, capture_path: Path) -> str:
    """Install capture after construction and finalize after measured runs."""

    capture_literal = repr(str(capture_path))
    install = textwrap.indent(
        textwrap.dedent(
            f"""
            from scripts.pr391_capture_choice_rows import ChoiceRowCapture
            import atexit as _capture_atexit
            import mtplx.fast_sampling as _capture_fast_sampling
            import mtplx.generation as _capture_generation

            _choice_capture = ChoiceRowCapture.install(
                _capture_fast_sampling,
                _capture_generation,
            )
            _capture_atexit.register(_choice_capture.close)
            _choice_capture_path = Path({capture_literal})
            """
        ).lstrip(),
        "    ",
    )
    source = _insert_once(source, INSTALL_ANCHOR, install)

    finalize = textwrap.indent(
        textwrap.dedent(
            """
            _capture_observed = {
                field: sum(int(row[field]) for row in rows)
                for field in (
                    "drafted_tokens",
                    "accepted_drafts",
                    "verify_calls",
                    "correction_tokens",
                    "bonus_tokens",
                )
            }
            _capture_expected = {
                'drafted_tokens': 3338,
                'accepted_drafts': 1656,
                'verify_calls': 1146,
                'correction_tokens': 789,
                'bonus_tokens': 342,
            }
            _capture_expected_rows = {
                20260829: {
                    "drafted_tokens": 1146,
                    "accepted_drafts": 566,
                    "verify_calls": 392,
                    "correction_tokens": 269,
                    "bonus_tokens": 119,
                    "response_token_sha256": (
                        "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc"
                    ),
                },
                20260830: {
                    "drafted_tokens": 1169,
                    "accepted_drafts": 576,
                    "verify_calls": 399,
                    "correction_tokens": 278,
                    "bonus_tokens": 117,
                    "response_token_sha256": (
                        "e50c3361a12a34d0b410819658acfc125e1559537434c57060de2cd90af94f16"
                    ),
                },
                20260831: {
                    "drafted_tokens": 1023,
                    "accepted_drafts": 514,
                    "verify_calls": 355,
                    "correction_tokens": 242,
                    "bonus_tokens": 106,
                    "response_token_sha256": (
                        "dc1816e3628d3e585ae4fe64d6745ca1c9f7ed30bde3128a624b3d5ae715e501"
                    ),
                },
            }
            for _capture_row in rows:
                _seed = int(_capture_row["seed"])
                _expected_row = _capture_expected_rows.get(_seed)
                _observed_row = {
                    field: _capture_row[field]
                    for field in (
                        "drafted_tokens",
                        "accepted_drafts",
                        "verify_calls",
                        "correction_tokens",
                        "bonus_tokens",
                        "response_token_sha256",
                    )
                }
                if _expected_row != _observed_row:
                    raise RuntimeError(
                        f"capture baseline row drifted for seed {_seed}: "
                        f"expected={_expected_row} observed={_observed_row}"
                    )
            _draft_misses = (
                _capture_observed["drafted_tokens"]
                - _capture_observed["accepted_drafts"]
            )
            _resolved_verify = (
                _capture_observed["bonus_tokens"]
                + _capture_observed["correction_tokens"]
            )
            _capture_hit_miss = {
                "draft_hits": _capture_observed["accepted_drafts"],
                "draft_misses": _draft_misses,
                "draft_hit_rate": (
                    _capture_observed["accepted_drafts"]
                    / _capture_observed["drafted_tokens"]
                ),
                "draft_miss_rate": (
                    _draft_misses / _capture_observed["drafted_tokens"]
                ),
                "verify_all_accepted_hits": _capture_observed["bonus_tokens"],
                "verify_rejection_correction_misses": (
                    _capture_observed["correction_tokens"]
                ),
                "verify_unresolved": (
                    _capture_observed["verify_calls"] - _resolved_verify
                ),
                "resolved_verify_hit_rate": (
                    _capture_observed["bonus_tokens"] / _resolved_verify
                ),
                "resolved_verify_miss_rate": (
                    _capture_observed["correction_tokens"] / _resolved_verify
                ),
            }
            _capture_depth_hit_miss = []
            for _capture_row in rows:
                _depth_rows = []
                for _depth, (_hits, _drafts) in enumerate(
                    zip(
                        _capture_row["accepted_by_depth"],
                        _capture_row["drafted_by_depth"],
                        strict=True,
                    )
                ):
                    _misses = int(_drafts) - int(_hits)
                    _depth_rows.append(
                        {
                            "depth": _depth,
                            "hits": int(_hits),
                            "misses": _misses,
                            "depth_hit_rate": int(_hits) / int(_drafts),
                            "depth_miss_rate": _misses / int(_drafts),
                        }
                    )
                _capture_depth_hit_miss.append(
                    {"seed": int(_capture_row["seed"]), "depths": _depth_rows}
                )
            _choice_capture.finalize(
                _choice_capture_path,
                metadata={
                    "capture_kind": "diagnostic_pre_top_p_draft_choices",
                    "driver_sha256": (
                        "0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0"
                    ),
                    "source_commit": source_commit,
                    "numpy_version": __import__("numpy").__version__,
                    "float32_policy": "benchmark_experiment_only_not_retainable",
                    "response_token_sha256": [
                        row["response_token_sha256"] for row in rows
                    ],
                    "accepted_by_depth": [row["accepted_by_depth"] for row in rows],
                    "drafted_by_depth": [row["drafted_by_depth"] for row in rows],
                    "online_correction_cache": [
                        row["online_correction_cache"] for row in rows
                    ],
                    "hit_miss": _capture_hit_miss,
                    "depth_hit_miss": _capture_depth_hit_miss,
                },
                expected_rows=3338,
                observed_counters=_capture_observed,
                expected_counters=_capture_expected,
            )
            _choice_capture.close()
            """
        ).lstrip(),
        "    ",
    )
    return _insert_once(source, FINALIZE_ANCHOR, finalize)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture-npz", type=Path, required=True)
    parser.add_argument("--reviewed-driver", type=Path, default=REVIEWED_DRIVER)
    return parser.parse_known_args(argv)


def validate_driver_argv(argv: Sequence[str]) -> argparse.Namespace:
    """Reject workload drift before the reviewed driver can load the model."""

    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-file", action="append", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--target-mode", required=True)
    parser.add_argument("--require-compiled-verify", action="store_true")
    parser.add_argument("--m4-stage3", action="store_true")
    parser.add_argument("--qsa-fused-kv-gather", action="store_true")
    parser.add_argument("--full-frspec", action="store_true")
    parser.add_argument("--compiled-mtp-prepare", action="store_true")
    parser.add_argument("--relaxed-draft-ties", action="store_true")
    parser.add_argument("--max-tokens", type=int, required=True)
    try:
        parsed, extras = parser.parse_known_args(list(argv))
    except argparse.ArgumentError as exc:
        raise DriverArgumentMismatch(f"noncanonical driver arguments: {exc}") from exc
    if extras:
        raise DriverArgumentMismatch(f"noncanonical extra driver arguments: {extras}")
    if parsed.source.resolve() != REVIEWED_SOURCE:
        raise DriverArgumentMismatch(f"source must be {REVIEWED_SOURCE}")
    if len(parsed.expected_source) != 40:
        raise DriverArgumentMismatch("expected-source must be a 40-character commit")
    if parsed.seed != list(REVIEWED_SEEDS):
        raise DriverArgumentMismatch(f"seeds must be {REVIEWED_SEEDS}")
    if parsed.target_mode != "batched":
        raise DriverArgumentMismatch("target-mode must be batched")
    if parsed.max_tokens != 1024:
        raise DriverArgumentMismatch("max-tokens must be 1024")
    required_flags = {
        "require_compiled_verify": parsed.require_compiled_verify,
        "m4_stage3": parsed.m4_stage3,
        "qsa_fused_kv_gather": parsed.qsa_fused_kv_gather,
        "full_frspec": parsed.full_frspec,
        "compiled_mtp_prepare": parsed.compiled_mtp_prepare,
        "relaxed_draft_ties": parsed.relaxed_draft_ties,
    }
    missing = sorted(name for name, enabled in required_flags.items() if not enabled)
    if missing:
        raise DriverArgumentMismatch(f"canonical flags missing: {missing}")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args, driver_argv = _parse_args(argv)
    validate_driver_argv(driver_argv)
    source = load_reviewed_driver(args.reviewed_driver)
    transformed = transform_capture_driver(source, capture_path=args.capture_npz)
    sys.argv = [str(args.reviewed_driver), *driver_argv]
    namespace = {
        "__name__": "__main__",
        "__file__": str(args.reviewed_driver),
        "__package__": None,
    }
    exec(compile(transformed, str(args.reviewed_driver), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
