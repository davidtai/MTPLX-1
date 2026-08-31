#!/usr/bin/env python3
"""Launch the reviewed PR #391 workload with one fixed NumPy choice arm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
from typing import Sequence

from scripts.pr391_capture_launcher import DriverSourceMismatch
from scripts.pr391_capture_launcher import REVIEWED_DRIVER
from scripts.pr391_capture_launcher import load_reviewed_driver
from scripts.pr391_capture_launcher import validate_driver_argv
from scripts.pr391_numpy_choice_routes import RouteArm


PRE_RUN_ANCHOR = "        started = time.perf_counter()"
ATTACH_ANCHOR = '        row["pre_run_reset"] = reset_receipt'


def _insert_once(source: str, anchor: str, insertion: str) -> str:
    if source.count(anchor) != 1:
        raise DriverSourceMismatch(
            f"reviewed driver anchor count changed for {anchor!r}"
        )
    return source.replace(anchor, insertion + anchor, 1)


def transform_choice_driver(source: str, *, arm: str | RouteArm) -> str:
    """Install one per-request fixed arm outside the measured timer."""

    try:
        installed_arm = RouteArm(arm)
    except ValueError as exc:
        raise ValueError(f"unknown PR391 choice-route arm: {arm!r}") from exc
    install = textwrap.indent(
        textwrap.dedent(
            f"""
            from scripts.pr391_numpy_choice_routes import NumpyChoiceRoute
            import atexit as _numpy_choice_atexit
            import mtplx.generation as _numpy_choice_generation

            _numpy_choice_route = NumpyChoiceRoute.install(
                _numpy_choice_generation,
                arm={installed_arm.value!r},
                expected_seed=seed,
            )
            _numpy_choice_atexit.register(_numpy_choice_route.close)
            """
        ).lstrip(),
        "        ",
    )
    source = _insert_once(source, PRE_RUN_ANCHOR, install)

    attach = textwrap.indent(
        textwrap.dedent(
            """
            _numpy_choice_route.close()
            _numpy_choice_receipt = _numpy_choice_route.finish_receipt(
                stats=output.stats
            )
            _draft_hits = int(row["accepted_drafts"])
            _draft_opportunities = int(row["drafted_tokens"])
            _draft_misses = _draft_opportunities - _draft_hits
            _resolved_verify = int(row["bonus_tokens"]) + int(
                row["correction_tokens"]
            )
            _depth_hit_miss = []
            for _depth, (_depth_hits, _depth_drafts) in enumerate(
                zip(
                    row["accepted_by_depth"],
                    row["drafted_by_depth"],
                    strict=True,
                )
            ):
                _depth_misses = int(_depth_drafts) - int(_depth_hits)
                _depth_hit_miss.append(
                    {
                        "depth": _depth,
                        "hits": int(_depth_hits),
                        "misses": _depth_misses,
                        "depth_hit_rate": int(_depth_hits) / int(_depth_drafts),
                        "depth_miss_rate": _depth_misses / int(_depth_drafts),
                    }
                )
            row["numpy_choice_route"] = _numpy_choice_receipt
            row["hit_miss"] = {
                "draft_hits": _draft_hits,
                "draft_misses": _draft_misses,
                "draft_hit_rate": _draft_hits / _draft_opportunities,
                "draft_miss_rate": _draft_misses / _draft_opportunities,
                "depths": _depth_hit_miss,
                "verify_all_accepted_hits": int(row["bonus_tokens"]),
                "verify_rejection_correction_misses": int(
                    row["correction_tokens"]
                ),
                "verify_unresolved": int(row["verify_calls"]) - _resolved_verify,
                "resolved_verify_hit_rate": (
                    int(row["bonus_tokens"]) / _resolved_verify
                ),
                "resolved_verify_miss_rate": (
                    int(row["correction_tokens"]) / _resolved_verify
                ),
            }
            """
        ).lstrip(),
        "        ",
    )
    return _insert_once(source, ATTACH_ANCHOR, attach)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--draft-choice-arm",
        choices=tuple(arm.value for arm in RouteArm),
        required=True,
    )
    parser.add_argument("--reviewed-driver", type=Path, default=REVIEWED_DRIVER)
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, driver_argv = _parse_args(argv)
    validate_driver_argv(driver_argv)
    source = load_reviewed_driver(args.reviewed_driver)
    transformed = transform_choice_driver(source, arm=args.draft_choice_arm)
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
