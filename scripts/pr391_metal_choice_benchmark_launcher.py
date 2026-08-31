#!/usr/bin/env python3
"""Launch the reviewed PR #391 workload with the exact Metal softfloat64 route."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.pr391_capture_launcher import DriverArgumentMismatch
from scripts.pr391_capture_launcher import DriverSourceMismatch
from scripts.pr391_capture_launcher import REVIEWED_DRIVER
from scripts.pr391_capture_launcher import load_reviewed_driver
from scripts.pr391_capture_launcher import validate_driver_argv


PREBIND_ANCHOR = "    after_load_memory = {"
PRE_RUN_ANCHOR = "        started = time.perf_counter()"
ATTACH_ANCHOR = '        row["pre_run_reset"] = reset_receipt'
THERMAL_GATE_ANCHOR = "        thermal_receipt = wait_for_temperature()"
ROUTE_ARM = "metal-softfloat64-exact-test-only"
REQUIRED_EXPECTED_FILES = frozenset(
    {
        "mtplx/kernels/_metal_softfloat64_v0_1_1.py",
        "mtplx/kernels/pr391_softfloat64_verifier_decision.py",
        "mtplx/generation.py",
        "mtplx/graphbank.py",
        "mtplx/models/qwen4_exp.py",
        "mtplx/pcg64_tape.py",
        "mtplx/pr391_mtp_handoff.py",
        "mtplx/qwen4_fixed_verify.py",
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


def prewarm_softfloat64_verifier_decision(
    mx: Any,
    verifier_kernel: Any,
) -> dict[str, Any]:
    """Compile and prove every verifier outcome class before request timing."""

    from mtplx.kernels.pr391_softfloat64_verifier_decision import (
        reference_pr391_softfloat64_verifier_decision,
    )

    def support_row(
        token: int,
        alternate: int,
        token_probability: float,
        *,
        filler_base: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.array(
            sorted([token, alternate, *range(filler_base, filler_base + 18)]),
            dtype=np.uint32,
        )
        probabilities = np.zeros(20, dtype=np.float32)
        values = np.full(20, np.float32(-1.0e30), dtype=np.float32)
        probabilities[np.flatnonzero(ids == token)[0]] = np.float32(
            token_probability
        )
        probabilities[np.flatnonzero(ids == alternate)[0]] = np.float32(
            1.0 - token_probability
        )
        positive = probabilities > np.float32(0.0)
        values[positive] = np.log(probabilities[positive]).astype(np.float32)
        return ids, values, probabilities

    draft_tokens = np.array([11, 22, 33], dtype=np.uint32)
    draft_rows = [
        support_row(11, 101, 0.5, filler_base=1000),
        support_row(22, 102, 0.5, filler_base=1100),
        support_row(33, 103, 0.5, filler_base=1200),
    ]
    target_rows = [
        support_row(11, 201, 0.25, filler_base=2000),
        support_row(22, 202, 0.25, filler_base=2100),
        support_row(33, 203, 0.25, filler_base=2200),
        support_row(301, 302, 0.25, filler_base=2300),
    ]
    draft_ids = np.stack([row[0] for row in draft_rows])
    draft_values = np.stack([row[1] for row in draft_rows])
    draft_probs = np.stack([row[2] for row in draft_rows])
    target_ids = np.stack([row[0] for row in target_rows])
    target_values = np.stack([row[1] for row in target_rows])
    target_probs = np.stack([row[2] for row in target_rows])
    stop_ids_default = np.array([999, 998], dtype=np.uint32)
    cases = (
        ("reject_d0", np.array([0.75, 0.50, 0.25, 0.50], np.float64), stop_ids_default, 0, True),
        ("reject_d1", np.array([0.25, 0.75, 0.50, 0.50], np.float64), stop_ids_default, 0, True),
        ("reject_d2", np.array([0.25, 0.25, 0.75, 0.50], np.float64), stop_ids_default, 0, True),
        ("all_accept_bonus", np.array([0.25, 0.25, 0.25, 0.50], np.float64), stop_ids_default, 0, True),
        ("accepted_stop", np.array([0.25, 0.25, 0.25, 0.50], np.float64), np.array([11, 998], np.uint32), 1, True),
        ("bonus_disabled", np.array([0.25, 0.25, 0.25, 0.50], np.float64), stop_ids_default, 0, False),
    )
    verified: list[str] = []
    for name, uniforms, stop_ids, stop_count, bonus_allowed in cases:
        expected = reference_pr391_softfloat64_verifier_decision(
            draft_tokens,
            draft_ids,
            draft_values,
            draft_probs,
            target_ids,
            target_values,
            target_probs,
            uniforms,
            stop_ids,
            stop_count=stop_count,
            bonus_allowed=bonus_allowed,
        )
        observed = tuple(
            verifier_kernel(
                mx.array(draft_tokens, dtype=mx.uint32),
                mx.array(draft_ids, dtype=mx.uint32),
                mx.array(draft_values, dtype=mx.float32),
                mx.array(draft_probs, dtype=mx.float32),
                mx.array(target_ids, dtype=mx.uint32),
                mx.array(target_values, dtype=mx.float32),
                mx.array(target_probs, dtype=mx.float32),
                mx.array(uniforms.view(np.uint64), dtype=mx.uint64),
                mx.array(stop_ids, dtype=mx.uint32),
                mx.array(np.array([stop_count], dtype=np.uint32), dtype=mx.uint32),
                mx.array(
                    np.array([int(bonus_allowed)], dtype=np.uint32),
                    dtype=mx.uint32,
                ),
            )
        )
        mx.eval(*observed)
        for output_index, (actual, wanted) in enumerate(
            zip(observed, expected, strict=True)
        ):
            actual_array = np.asarray(actual)
            if actual_array.dtype != wanted.dtype or actual_array.shape != wanted.shape:
                raise RuntimeError(
                    f"PR391 verifier {name} output {output_index} ABI mismatch"
                )
            if not np.array_equal(actual_array, wanted):
                raise RuntimeError(
                    f"PR391 verifier {name} output {output_index} parity mismatch: "
                    f"expected={wanted.tolist()} observed={actual_array.tolist()}"
                )
        verified.append(name)
    return {
        "status": "passed",
        "cases": verified,
        "case_count": len(verified),
        "reference": "literal_numpy_2.4.4_float64",
    }


def _pcg64_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class PreboundSoftFloat64ChoiceKernel:
    """Construction-bound exact selector and its guarded prewarm evidence."""

    mx: Any
    selector: Any
    source_sha256: Mapping[str, str]
    prewarm_receipt: Mapping[str, Any]


def prebind_softfloat64_choice_kernel() -> PreboundSoftFloat64ChoiceKernel:
    """Bind and prove the exact raw-K20 selector before request timing."""

    import mlx.core as mx

    from mtplx.kernels._metal_softfloat64_v0_1_1 import (
        METAL_SOFTFLOAT_COMMIT,
        METAL_SOFTFLOAT_SOURCE,
        METAL_SOFTFLOAT_VERSION,
    )
    from mtplx.kernels.pr391_softfloat64_verifier_decision import (
        METAL_CONTROLLER_SOURCE,
        bind_pr391_softfloat64_candidate_selector,
        reference_select_candidate_row,
    )

    selector = bind_pr391_softfloat64_candidate_selector()
    ids_host = np.array(
        [[109, 101, 117, 104, 115, 103, 119, 102, 118, 105,
          116, 106, 114, 107, 113, 108, 112, 110, 111, 100]],
        dtype=np.uint32,
    )
    values_host = np.linspace(3.0, -3.0, 20, dtype=np.float32).reshape(1, 20)
    probs_host = np.exp(values_host).astype(np.float32)
    probs_host /= np.float32(np.sum(probs_host, dtype=np.float32))
    uniform = np.array([np.ldexp(np.float64(391), -10)], dtype=np.float64)
    expected, _, _ = reference_select_candidate_row(
        ids_host[0], values_host[0], probs_host[0], uniform[0], top_p=0.95
    )
    observed = tuple(
        selector(
            mx.array(ids_host, dtype=mx.uint32),
            mx.array(values_host, dtype=mx.float32),
            mx.array(probs_host, dtype=mx.float32),
            mx.array(uniform.view(np.uint64), dtype=mx.uint64),
        )
    )
    mx.eval(*observed)
    selected = np.asarray(observed[0], dtype=np.uint32)
    raw_ids = np.asarray(observed[1], dtype=np.uint32)
    raw_values = np.asarray(observed[2], dtype=np.float32)
    raw_probs = np.asarray(observed[3], dtype=np.float32)
    selected_match = selected.shape == (1,) and int(selected[0]) == expected
    raw_exact = (
        np.array_equal(raw_ids, ids_host)
        and np.array_equal(raw_values.view(np.uint32), values_host.view(np.uint32))
        and np.array_equal(raw_probs.view(np.uint32), probs_host.view(np.uint32))
    )
    if not selected_match or not raw_exact:
        raise RuntimeError("softfloat64 selector prewarm parity failed")
    hashes = {
        "metal_softfloat64": hashlib.sha256(
            METAL_SOFTFLOAT_SOURCE.encode()
        ).hexdigest(),
        "controller": hashlib.sha256(METAL_CONTROLLER_SOURCE.encode()).hexdigest(),
    }
    receipt = {
        "status": "passed",
        "rows": 1,
        "raw_passthrough_bit_exact": True,
        "selected_token_match": True,
        "schedule_id": "softfloat64-k20-top-p-0.95-numpy-2.4.4",
        "metal_softfloat_version": METAL_SOFTFLOAT_VERSION,
        "metal_softfloat_commit": METAL_SOFTFLOAT_COMMIT,
    }
    return PreboundSoftFloat64ChoiceKernel(
        mx=mx,
        selector=selector,
        source_sha256=hashes,
        prewarm_receipt=receipt,
    )


class PR391DirectSoftFloat64D3Route:
    """Benchmark-only owner of the exact joint-D3 PCG64 uniform tape."""

    def __init__(
        self,
        generation_module: Any,
        *,
        expected_seed: int,
        max_output_tokens: int,
        kernel_module: Any,
        verifier_kernel: Any,
        verifier_prewarm: Mapping[str, Any],
        target_sampler: Any,
        draft_sampler: Any,
    ) -> None:
        from mtplx.pcg64_tape import PCG64UniformTape
        if not isinstance(kernel_module, PreboundSoftFloat64ChoiceKernel):
            raise RuntimeError("joint D3 requires PreboundSoftFloat64ChoiceKernel")
        if kernel_module.prewarm_receipt is None:
            raise RuntimeError("joint D3 requires the prewarmed Metal selector")
        if not callable(verifier_kernel):
            raise RuntimeError("joint D3 requires the prewarmed verifier decision")
        if (
            verifier_prewarm.get("status") != "passed"
            or int(verifier_prewarm.get("case_count", -1)) != 6
        ):
            raise RuntimeError("joint D3 requires six verifier parity prewarm cases")
        for sampler in (target_sampler, draft_sampler):
            if (
                float(sampler.temperature) != 1.0
                or int(sampler.top_k) != 20
                or float(sampler.top_p) != 0.95
            ):
                raise RuntimeError(
                    "joint D3 requires temperature=1, top_k=20, top_p=0.95"
                )
        if isinstance(expected_seed, bool) or isinstance(max_output_tokens, bool):
            raise TypeError("joint D3 seed and maximum output must be integers")
        self.expected_seed = int(expected_seed)
        self.max_output_tokens = int(max_output_tokens)
        if not 0 < self.max_output_tokens <= 16_384:
            raise ValueError("joint D3 maximum output must be in [1, 16384]")

        request_rng = np.random.default_rng(self.expected_seed)
        self._start_hash = _pcg64_state_sha256(request_rng)
        self.uniform_tape = PCG64UniformTape.build(
            request_rng,
            max_output_tokens=self.max_output_tokens,
        )
        uniform_bits = np.ascontiguousarray(
            self.uniform_tape.device_values,
            dtype=np.float64,
        ).view(np.uint64)
        self.uniform_bit_rows = kernel_module.mx.array(
            uniform_bits,
            dtype=kernel_module.mx.uint64,
        )
        self.bonus_allowed_rows = (
            kernel_module.mx.array(np.array([0], dtype=np.uint32), dtype=kernel_module.mx.uint32),
            kernel_module.mx.array(np.array([1], dtype=np.uint32), dtype=kernel_module.mx.uint32),
        )
        self.generation_module = generation_module
        self.prebound_kernel = kernel_module
        self.verifier_kernel = verifier_kernel
        self.verifier_prewarm = dict(verifier_prewarm)
        self.sampler = target_sampler
        self.draft_sampler = draft_sampler
        self.preserve_paged = True
        self.claimed = False
        self._installed = False
        self._closed = False

    @classmethod
    def install(
        cls,
        generation_module: Any,
        *,
        expected_seed: int,
        max_output_tokens: int,
        kernel_module: Any,
        verifier_kernel: Any,
        verifier_prewarm: Mapping[str, Any],
        target_sampler: Any,
        draft_sampler: Any,
    ) -> "PR391DirectSoftFloat64D3Route":
        route = cls(
            generation_module,
            expected_seed=expected_seed,
            max_output_tokens=max_output_tokens,
            kernel_module=kernel_module,
            verifier_kernel=verifier_kernel,
            verifier_prewarm=verifier_prewarm,
            target_sampler=target_sampler,
            draft_sampler=draft_sampler,
        )
        generation_module._pr391_install_float32_d3_request_route(route)
        route._installed = True
        return route

    def close(self) -> None:
        if not self._installed:
            return
        self.generation_module._pr391_uninstall_float32_d3_request_route(self)
        self._installed = False
        self._closed = True

    def finish_receipt(
        self,
        *,
        stats: Mapping[str, int] | Any,
    ) -> dict[str, Any]:
        if not self._closed or not self.claimed:
            raise RuntimeError("joint D3 receipt requires one claimed and closed route")
        normalized = {
            field: int(
                stats[field] if isinstance(stats, Mapping) else getattr(stats, field)
            )
            for field in COUNTER_FIELDS
        }
        normalized["drafted_by_depth"] = [
            int(value)
            for value in (
                stats["drafted_by_depth"]
                if isinstance(stats, Mapping)
                else stats.drafted_by_depth
            )
        ]
        route_counts = build_d3_route_counts(stats)
        drafted = route_counts["d3_rows"]
        prewarm = self.prebound_kernel.prewarm_receipt
        return {
            "schema_version": 2,
            "receipt_kind": "final_success",
            "arm": ROUTE_ARM,
            "expected_seed": self.expected_seed,
            "start_pcg64_state_sha256": self._start_hash,
            "final_pcg64_state_sha256": _pcg64_state_sha256(self.uniform_tape.rng),
            "route_counts": {
                "calls": drafted,
                "matched_rows": drafted,
                "raw_passthrough_rows": drafted,
                "pending": 0,
                "failures": 0,
                "count_source": "stats.events.draft_core",
                **route_counts,
            },
            "prebound": dict(prewarm or {}),
            "verifier_prebound": dict(self.verifier_prewarm),
            "source_sha256": dict(self.prebound_kernel.source_sha256),
            "schedule": {
                "uniform": "one_request_pcg64_tape_three_draw_joint_d3",
                "uniform_bit_rows": int(self.uniform_tape.device_values.size),
                "uniform_device_installs": 1,
                "kernel_top_p": 0.95,
            },
            "policy": {
                "fixed_at_install": True,
                "evaluation_scope": "benchmark_experiment_only",
                "retention_eligible": False,
                "sync_boundary": "one_materialization_per_full_d3_chain",
            },
            "stats": normalized,
        }


def build_d3_route_counts(stats: Mapping[str, Any] | Any) -> dict[str, int | str]:
    """Prove fixed D3 engagement from construction and existing depth ledgers."""

    def value(name: str) -> Any:
        return stats[name] if isinstance(stats, Mapping) else getattr(stats, name)

    drafted_by_depth = [int(item) for item in value("drafted_by_depth")]
    if (
        len(drafted_by_depth) != 3
        or drafted_by_depth[0] <= 0
        or not (
            drafted_by_depth[0] >= drafted_by_depth[1] >= drafted_by_depth[2] > 0
        )
    ):
        raise RuntimeError(
            f"PR391 fixed D3 depth ledger is invalid: {drafted_by_depth}"
        )
    d3_cycles = drafted_by_depth[2]
    shortened_d2_cycles = drafted_by_depth[1] - drafted_by_depth[2]
    shortened_d1_cycles = drafted_by_depth[0] - drafted_by_depth[1]
    d3_rows = 3 * d3_cycles
    other_draft_rows = 2 * shortened_d2_cycles + shortened_d1_cycles
    if int(value("drafted_tokens")) != d3_rows + other_draft_rows:
        raise RuntimeError(
            "PR391 fixed D3 aggregate does not match its depth ledger: "
            f"drafted_tokens={value('drafted_tokens')} depths={drafted_by_depth}"
        )
    return {
        "d3_cycles": d3_cycles,
        "d3_rows": d3_rows,
        "shortened_d2_cycles": shortened_d2_cycles,
        "shortened_d1_cycles": shortened_d1_cycles,
        "context_copy_rounds": int(value("context_copy_rounds")),
        "context_copy_drafted_tokens": int(value("context_copy_drafted_tokens")),
        "other_draft_rows": other_draft_rows,
        "count_source": "construction_claim_plus_stats.drafted_by_depth",
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


def build_exact_output_parity_receipt(
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
    route = row.get("metal_softfloat64_choice_route")
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
        "arithmetic": "metal-softfloat64-exact",
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


def validate_metal_choice_receipt(receipt: Mapping[str, Any]) -> None:
    """Fail closed on exact route integrity."""

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
    verifier_prebound = receipt.get("verifier_prebound")
    if not isinstance(verifier_prebound, Mapping):
        raise RuntimeError("Metal choice route contract missing verifier prebind")
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
    if (
        verifier_prebound.get("status") != "passed"
        or int(verifier_prebound.get("case_count", -1)) != 6
    ):
        raise RuntimeError(
            "Metal choice route contract failed verifier prebind: "
            f"{dict(verifier_prebound)}"
        )
    stats = receipt.get("stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError("Metal choice route contract missing stats")
    expected_rows = int(counts.get("d3_rows", -1))
    d3_cycles = int(counts.get("d3_cycles", -1))
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
        expected_rows <= 0
        or d3_cycles <= 0
        or expected_rows != 3 * d3_cycles
        or int(stats.get("drafted_tokens", -1))
        != expected_rows + int(counts.get("other_draft_rows", -1))
        or observed["calls"] != expected_rows
        or observed["matched_rows"] != expected_rows
        or observed["raw_passthrough_rows"] != expected_rows
        or observed["pending"] != 0
        or observed["failures"] != 0
    ):
        raise RuntimeError(
            "Metal choice route contract failed: "
            f"d3_rows={expected_rows} route_counts={dict(counts)}"
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


def transform_metal_choice_driver(source: str, *, retain_events: bool = False) -> str:
    """Install the exact Metal softfloat64 route outside measured intervals."""

    if source.count(THERMAL_GATE_ANCHOR) != 1:
        raise DriverSourceMismatch("reviewed driver thermal-gate anchor changed")
    source = source.replace(
        THERMAL_GATE_ANCHOR,
        '        thermal_receipt = {"disabled": True, "reason": "user_requested"}',
        1,
    )

    retain_events_setup = (
        'import os as _metal_choice_os\n'
        '_metal_choice_os.environ["MTPLX_DROP_EVENTS"] = "0"\n'
        if retain_events
        else ""
    )
    prebind = textwrap.indent(
        retain_events_setup
        + textwrap.dedent(
            """
            from scripts.pr391_metal_choice_benchmark_launcher import build_exact_output_parity_receipt
            from scripts.pr391_metal_choice_benchmark_launcher import build_hit_miss_receipt
            from scripts.pr391_metal_choice_benchmark_launcher import prewarm_softfloat64_verifier_decision
            from scripts.pr391_metal_choice_benchmark_launcher import PR391DirectSoftFloat64D3Route
            from scripts.pr391_metal_choice_benchmark_launcher import prebind_softfloat64_choice_kernel
            from scripts.pr391_metal_choice_benchmark_launcher import validate_metal_choice_receipt
            from mtplx.kernels.pr391_softfloat64_verifier_decision import bind_pr391_softfloat64_verifier_decision
            import atexit as _metal_choice_atexit
            import mtplx.generation as _metal_choice_generation

            _metal_choice_prebound = prebind_softfloat64_choice_kernel()
            _metal_choice_verifier = bind_pr391_softfloat64_verifier_decision()
            _metal_choice_verifier_prewarm = prewarm_softfloat64_verifier_decision(
                _metal_choice_prebound.mx,
                _metal_choice_verifier,
            )
            """
        ).lstrip(),
        "    ",
    )
    source = _insert_once(source, PREBIND_ANCHOR, prebind)

    install = textwrap.indent(
        textwrap.dedent(
            """
            _metal_choice_route = PR391DirectSoftFloat64D3Route.install(
                _metal_choice_generation,
                expected_seed=seed,
                max_output_tokens=max_tokens,
                kernel_module=_metal_choice_prebound,
                verifier_kernel=_metal_choice_verifier,
                verifier_prewarm=_metal_choice_verifier_prewarm,
                target_sampler=sampler,
                draft_sampler=cell.get("draft_sampler", sampler),
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
                stats=output.stats,
            )
            validate_metal_choice_receipt(
                _metal_choice_receipt,
            )
            row["metal_softfloat64_choice_route"] = _metal_choice_receipt
            row["response_token_ids"] = [int(token) for token in output.tokens]
            row["softfloat64_output_parity"] = build_exact_output_parity_receipt(row)
            print(
                "[pr391-softfloat64] result "
                + json.dumps(
                    {
                        "seed": int(row["seed"]),
                        "decode_elapsed_s": float(row["decode_elapsed_s"]),
                        "decode_tok_s": float(row["decode_tok_s"]),
                        "parity": row["softfloat64_output_parity"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not row["softfloat64_output_parity"]["exact_reference_match"]:
                _metal_choice_failure_path = OUT_DIR / (
                    f"pr391-softfloat64-parity-failure-{sequence}-{seed}.json"
                )
                _metal_choice_failure_path.write_text(
                    json.dumps(
                        {
                            "seed": int(seed),
                            "sequence": int(sequence),
                            "response_token_ids": row["response_token_ids"],
                            "generation_events": list(output.stats.events),
                            "parity": row["softfloat64_output_parity"],
                        },
                        sort_keys=True,
                    )
                )
                print(
                    "[pr391-softfloat64] parity diagnostic "
                    + str(_metal_choice_failure_path),
                    flush=True,
                )
                raise RuntimeError("Metal softfloat64 output or PCG64 parity failed")
            row["hit_miss"] = build_hit_miss_receipt(row)
            """
        ).lstrip(),
        "        ",
    )
    return _insert_once(source, ATTACH_ANCHOR, attach)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reviewed-driver", type=Path, default=REVIEWED_DRIVER)
    parser.add_argument("--retain-events", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, driver_argv = _parse_args(argv)
    driver_args = validate_driver_argv(driver_argv)
    validate_candidate_file_gates(driver_args.expected_file)
    source = load_reviewed_driver(args.reviewed_driver)
    transformed = transform_metal_choice_driver(
        source,
        retain_events=args.retain_events,
    )
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
