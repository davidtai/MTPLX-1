#!/usr/bin/env python3
"""Launch the reviewed PR #391 workload with the test-only Metal f32 route."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
from typing import Any, Mapping, Sequence

from scripts.pr391_capture_launcher import DriverArgumentMismatch
from scripts.pr391_capture_launcher import DriverSourceMismatch
from scripts.pr391_capture_launcher import REVIEWED_DRIVER
from scripts.pr391_capture_launcher import load_reviewed_driver
from scripts.pr391_capture_launcher import validate_driver_argv


PREBIND_ANCHOR = "    after_load_memory = {"
PRE_RUN_ANCHOR = "        started = time.perf_counter()"
ATTACH_ANCHOR = '        row["pre_run_reset"] = reset_receipt'
ROUTE_ARM = "metal-float32-test-only"
REQUIRED_EXPECTED_FILES = frozenset(
    {
        "mtplx/kernels/qwen4_frspec_k20_float32_choice.py",
        "scripts/pr391_metal_choice_benchmark_launcher.py",
        "scripts/pr391_metal_choice_route.py",
    }
)

COUNTER_FIELDS = (
    "drafted_tokens",
    "accepted_drafts",
    "verify_calls",
    "correction_tokens",
    "bonus_tokens",
)

REFERENCE_ROWS: dict[int, dict[str, Any]] = {
    20260829: {
        "response_token_sha256": (
            "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc"
        ),
        "drafted_tokens": 1146,
        "accepted_drafts": 566,
        "verify_calls": 392,
        "correction_tokens": 269,
        "bonus_tokens": 119,
        "accepted_by_depth": [259, 187, 120],
        "drafted_by_depth": [382, 382, 382],
        "start_pcg64_state_sha256": "da7887652f7f11899e362b48661512dcf59e404f9b6bd58720f93095ccccdb3e",
        "final_pcg64_state_sha256": "d291d8dcea3fcd76576a911b34c194ee03c740bb70b38c7d911ac1b8edce6f65",
    },
    20260830: {
        "response_token_sha256": (
            "e50c3361a12a34d0b410819658acfc125e1559537434c57060de2cd90af94f16"
        ),
        "drafted_tokens": 1169,
        "accepted_drafts": 576,
        "verify_calls": 399,
        "correction_tokens": 278,
        "bonus_tokens": 117,
        "accepted_by_depth": [275, 184, 117],
        "drafted_by_depth": [390, 390, 389],
        "start_pcg64_state_sha256": "a26d78c777fa1b3582c057531e4e726318a9fe9ac0291e9730b866bb205fa695",
        "final_pcg64_state_sha256": "9c8e99900ff4bf19d269a16e5f96ea38ae1104fdf28f604af576cc47941bd09f",
    },
    20260831: {
        "response_token_sha256": (
            "dc1816e3628d3e585ae4fe64d6745ca1c9f7ed30bde3128a624b3d5ae715e501"
        ),
        "drafted_tokens": 1023,
        "accepted_drafts": 514,
        "verify_calls": 355,
        "correction_tokens": 242,
        "bonus_tokens": 106,
        "accepted_by_depth": [245, 163, 106],
        "drafted_by_depth": [341, 341, 341],
        "start_pcg64_state_sha256": "f5c4a94b560cdd3ad78551b841f82b48331620852dd0763aaa84d07940fc0747",
        "final_pcg64_state_sha256": "7b9a66f542147dcd0f5576d7eeb56a81a320488a6ab57925dc6eb2beb1d06dfb",
    },
}


def _insert_once(source: str, anchor: str, insertion: str) -> str:
    if source.count(anchor) != 1:
        raise DriverSourceMismatch(
            f"reviewed driver anchor count changed for {anchor!r}"
        )
    return source.replace(anchor, insertion + anchor, 1)


def build_hit_miss_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build post-timer draft-depth and resolved-verifier rates."""

    draft_hits = int(row["accepted_drafts"])
    draft_opportunities = int(row["drafted_tokens"])
    draft_misses = draft_opportunities - draft_hits
    depths = []
    for depth, (hits_value, drafts_value) in enumerate(
        zip(
            row["accepted_by_depth"],
            row["drafted_by_depth"],
            strict=True,
        )
    ):
        hits = int(hits_value)
        drafts = int(drafts_value)
        misses = drafts - hits
        depths.append(
            {
                "depth": depth,
                "hits": hits,
                "misses": misses,
                "depth_hit_rate": hits / drafts,
                "depth_miss_rate": misses / drafts,
            }
        )

    bonus = int(row["bonus_tokens"])
    correction = int(row["correction_tokens"])
    resolved_verify = bonus + correction
    return {
        "draft_hits": draft_hits,
        "draft_misses": draft_misses,
        "draft_hit_rate": draft_hits / draft_opportunities,
        "draft_miss_rate": draft_misses / draft_opportunities,
        "depths": depths,
        "verify_all_accepted_hits": bonus,
        "verify_rejection_correction_misses": correction,
        "verify_unresolved": int(row["verify_calls"]) - resolved_verify,
        "resolved_verify_hit_rate": bonus / resolved_verify,
        "resolved_verify_miss_rate": correction / resolved_verify,
    }


def build_float32_output_drift_receipt(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one candidate row with the corrected exact control reference."""

    seed = int(row["seed"])
    try:
        reference = REFERENCE_ROWS[seed]
    except KeyError as exc:
        raise RuntimeError(f"no corrected PR391 reference for seed {seed}") from exc

    expected = {
        "response_token_sha256": str(reference["response_token_sha256"]),
        **{field: int(reference[field]) for field in COUNTER_FIELDS},
        "accepted_by_depth": [int(value) for value in reference["accepted_by_depth"]],
        "drafted_by_depth": [int(value) for value in reference["drafted_by_depth"]],
        "start_pcg64_state_sha256": str(reference["start_pcg64_state_sha256"]),
        "final_pcg64_state_sha256": str(reference["final_pcg64_state_sha256"]),
    }
    route = row.get("metal_float32_choice_route")
    if not isinstance(route, Mapping):
        raise RuntimeError("Metal choice route receipt is required for RNG drift")
    observed = {
        "response_token_sha256": str(row["response_token_sha256"]),
        **{field: int(row[field]) for field in COUNTER_FIELDS},
        "accepted_by_depth": [int(value) for value in row["accepted_by_depth"]],
        "drafted_by_depth": [int(value) for value in row["drafted_by_depth"]],
        "start_pcg64_state_sha256": str(route["start_pcg64_state_sha256"]),
        "final_pcg64_state_sha256": str(route["final_pcg64_state_sha256"]),
    }
    counter_deltas = {
        field: observed[field] - expected[field] for field in COUNTER_FIELDS
    }
    accepted_by_depth_deltas = [
        candidate - control
        for candidate, control in zip(
            observed["accepted_by_depth"],
            expected["accepted_by_depth"],
            strict=True,
        )
    ]
    drafted_by_depth_deltas = [
        candidate - control
        for candidate, control in zip(
            observed["drafted_by_depth"],
            expected["drafted_by_depth"],
            strict=True,
        )
    ]
    digest_match = (
        observed["response_token_sha256"] == expected["response_token_sha256"]
    )
    counters_match = all(delta == 0 for delta in counter_deltas.values())
    depths_match = all(
        delta == 0 for delta in (*accepted_by_depth_deltas, *drafted_by_depth_deltas)
    )
    rng_state_match = (
        observed["start_pcg64_state_sha256"] == expected["start_pcg64_state_sha256"]
        and observed["final_pcg64_state_sha256"] == expected["final_pcg64_state_sha256"]
    )
    token_level_severity = (
        {
            "status": "zero_drift_by_exact_1024_token_digest",
            "first_divergence_index": None,
            "differing_positions": 0,
            "edit_distance": 0,
        }
        if digest_match
        else {
            "status": "reference_token_ids_required",
            "first_divergence_index": None,
            "differing_positions": None,
            "edit_distance": None,
        }
    )
    return {
        "reference_kind": "corrected-variable-length-m4-exact-float64",
        "float32_policy": "benchmark-experiment-only-not-retainable",
        "seed": seed,
        "expected": expected,
        "observed": observed,
        "digest_match": digest_match,
        "counter_deltas": counter_deltas,
        "counters_match": counters_match,
        "accepted_by_depth_deltas": accepted_by_depth_deltas,
        "drafted_by_depth_deltas": drafted_by_depth_deltas,
        "depths_match": depths_match,
        "rng_state_match": rng_state_match,
        "token_level_severity": token_level_severity,
        "exact_reference_match": (
            digest_match and counters_match and depths_match and rng_state_match
        ),
    }


def validate_metal_choice_receipt(
    receipt: Mapping[str, Any], *, drafted_tokens: int
) -> None:
    """Fail closed on route integrity while permitting candidate output drift."""

    if receipt.get("arm") != ROUTE_ARM:
        raise RuntimeError(
            f"Metal choice route arm mismatch: expected {ROUTE_ARM!r}, "
            f"got {receipt.get('arm')!r}"
        )
    counts = receipt.get("route_counts")
    if not isinstance(counts, Mapping):
        raise RuntimeError("Metal choice route contract missing route_counts")
    prebound = receipt.get("prebound")
    if not isinstance(prebound, Mapping):
        raise RuntimeError("Metal choice route contract missing prebound receipt")
    if (
        prebound.get("status") != "passed"
        or int(prebound.get("rows", -1)) != 1
        or prebound.get("raw_passthrough_bit_exact") is not True
        or prebound.get("selected_token_match") is not True
        or not isinstance(prebound.get("schedule_id"), str)
        or not prebound["schedule_id"]
    ):
        raise RuntimeError(
            f"Metal choice route contract failed prebind: {dict(prebound)}"
        )
    expected_rows = int(drafted_tokens)
    observed = {
        field: int(counts.get(field, -1))
        for field in (
            "calls",
            "matched_rows",
            "raw_passthrough_rows",
            "pending",
            "failures",
        )
    }
    if (
        observed["calls"] != expected_rows
        or observed["matched_rows"] != expected_rows
        or observed["raw_passthrough_rows"] != expected_rows
        or observed["pending"] != 0
        or observed["failures"] != 0
    ):
        raise RuntimeError(
            "Metal choice route contract failed: "
            f"drafted_tokens={expected_rows} route_counts={dict(counts)}"
        )


def validate_candidate_file_gates(expected_files: Sequence[str]) -> None:
    """Require hashes for both experimental sources before model loading."""

    paths: set[str] = set()
    for spec in expected_files:
        try:
            path, digest = spec.split("=", 1)
        except ValueError as exc:
            raise DriverArgumentMismatch(
                f"bad --expected-file value: {spec!r}"
            ) from exc
        if path in paths or len(digest) != 64:
            raise DriverArgumentMismatch(
                f"bad or duplicate --expected-file value: {spec!r}"
            )
        paths.add(path)
    missing = sorted(REQUIRED_EXPECTED_FILES - paths)
    if missing:
        raise DriverArgumentMismatch(
            f"required --expected-file gates missing: {missing}"
        )


def transform_metal_choice_driver(source: str) -> str:
    """Install the fixed Metal f32 route outside each measured interval."""

    prebind = textwrap.indent(
        textwrap.dedent(
            """
            from scripts.pr391_metal_choice_benchmark_launcher import build_float32_output_drift_receipt
            from scripts.pr391_metal_choice_benchmark_launcher import build_hit_miss_receipt
            from scripts.pr391_metal_choice_benchmark_launcher import validate_metal_choice_receipt
            from scripts.pr391_metal_choice_route import MetalFloat32ChoiceRoute
            from scripts.pr391_metal_choice_route import prebind_metal_float32_choice_kernel
            import atexit as _metal_choice_atexit
            import mtplx.generation as _metal_choice_generation

            _metal_choice_prebound = prebind_metal_float32_choice_kernel()
            _metal_choice_prewarm = _metal_choice_prebound.prewarm_b1()
            """
        ).lstrip(),
        "    ",
    )
    source = _insert_once(source, PREBIND_ANCHOR, prebind)

    install = textwrap.indent(
        textwrap.dedent(
            """
            _metal_choice_route = MetalFloat32ChoiceRoute.install(
                _metal_choice_generation,
                expected_seed=seed,
                kernel_module=_metal_choice_prebound,
                sampler=cell.get("draft_sampler", sampler),
            )
            _metal_choice_atexit.register(_metal_choice_route.close)
            """
        ).lstrip(),
        "        ",
    )
    source = _insert_once(source, PRE_RUN_ANCHOR, install)

    attach = textwrap.indent(
        textwrap.dedent(
            """
            _metal_choice_route.close()
            _metal_choice_receipt = _metal_choice_route.finish_receipt(
                stats=output.stats
            )
            validate_metal_choice_receipt(
                _metal_choice_receipt,
                drafted_tokens=int(row["drafted_tokens"]),
            )
            row["metal_float32_choice_route"] = _metal_choice_receipt
            row["response_token_ids"] = [int(token) for token in output.tokens]
            row["float32_output_drift"] = build_float32_output_drift_receipt(row)
            if (
                row["float32_output_drift"]["digest_match"]
                and not row["float32_output_drift"]["rng_state_match"]
            ):
                raise RuntimeError(
                    "Metal float32 preserved output but drifted the PCG64 cursor"
                )
            row["hit_miss"] = build_hit_miss_receipt(row)
            """
        ).lstrip(),
        "        ",
    )
    return _insert_once(source, ATTACH_ANCHOR, attach)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reviewed-driver", type=Path, default=REVIEWED_DRIVER)
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, driver_argv = _parse_args(argv)
    driver_args = validate_driver_argv(driver_argv)
    validate_candidate_file_gates(driver_args.expected_file)
    source = load_reviewed_driver(args.reviewed_driver)
    transformed = transform_metal_choice_driver(source)
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
