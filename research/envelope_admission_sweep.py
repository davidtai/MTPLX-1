#!/usr/bin/env python3
"""CPU-only admission-gate sweep over the hy3-oq2e-rq4 envelope matrix.

Context: the 88x16k envelope-matrix cell (evals/tier2/oq2e_rq4_88_16k.{json,log},
NOTES.md entry "T3 envelope-matrix cell 88x16k", commit 29535e84) failed the
runtime's config-time admission gate with `load_count=0` -- a genuine
config-time rejection, zero MLX allocation, zero GPU touch. This script
re-exercises that SAME gate for the full {preset} x {max-live-kv-tokens}
matrix, entirely on CPU: it drives the real preset/argparse machinery from
scripts/benchmark_q2_mtp_depth_matrix.py to build the exact
mtplx.expert_runtime.ExpertStreamingConfig a real run would build, then
replicates the CPU-only prefix of mtplx/runtime.py's `_load_impl`
(the block that computes `additional_resident_bytes` from the externally
resident bf16 MTP head + the non-stock splitk router kernel's incremental
bytes, then calls `config.memory_plan(...)` and checks `plan.fits_fixed`)
-- stopping BEFORE `apply_mlx_memory_cap` / `ExpertStreamingRuntime.open`,
so no MLX buffer is ever allocated and no weight bytes are ever read.

Gate exercised (parity, not a re-implementation): mtplx/runtime.py, the
`if not streaming_plan.fits_fixed: raise ExpertStreamingConfigurationError(...)`
block inside `_load_impl` (~line 698-702), which is IDENTICAL in error class
and message template to the internal duplicate gate inside
`ExpertStreamingRuntime.open` (mtplx/expert_runtime.py ~line 2024-2027):

    f"fixed expert-streaming footprint exceeds limit by {-plan.unallocated_bytes} bytes"

The `_load_impl` gate is the one that actually fired for the 88x16k cell
(load_count=0 means `ExpertStreamingRuntime.open` was never reached), so this
script reconstructs THAT exact preflight, not the later duplicate.

Per REJECT cell, also evaluates a memory-limit override that adds exactly the
KV-token delta over the preset's max-live-kv-tokens default (4096), at
`kv_bytes_per_token` for the hy3-oq2e model spec (327_680 B/token -- the
envelope-accounting rule: envelope is the WEIGHTS budget, KV stacks on top;
the override never touches island count or cache sizing):

    override_bytes    = (kv_tokens - 4096) * 327_680
    override_limit     = preset_memory_limit_bytes + override_bytes

A cell that still rejects after this override indicates the PRESET's own
fixed footprint over-books its declared memory-limit independent of KV
(as observed for hy3-oq2e-rq4-88, which over-books by ~0.6 GiB even at its
own 4096-token default).

Usage:
    PYTHONPATH=<worktree> python3 research/envelope_admission_sweep.py \\
        [--output PATH]

No GPU, no MLX allocation, no weight loads. Only file reads: config.json
(small JSON), expert-manifest.json (structural JSON, not weight bytes), and
the MTP artifact's safetensors HEADER (via open_verified_hy3_mtp_artifacts,
which reads only the header via os.pread and never calls mx.load).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_q2_mtp_depth_matrix as bqmdm  # noqa: E402

from mtplx.artifacts import load_config  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    parse_memory_bytes,
    proj_quant_plan_discount,
    proj_requant_plan_discount,
    resolve_island_placement,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.hy3_mtp_patch import open_verified_hy3_mtp_artifacts  # noqa: E402
from mtplx.models.hy3_mlx import (  # noqa: E402
    estimate_hy3_router_kernel_incremental_bytes,
)

GIB = 1024**3
# HY3_EXPERT_OQ2E.kv_bytes_per_token (mtplx/expert_streaming_models.py) -- the
# envelope-accounting constant: bytes of KV cache added per live token,
# independent of island/cache sizing.
KV_BYTES_PER_TOKEN = 327_680
CONTROL_KV_TOKENS = 4096

PRESETS = (
    "hy3-oq2e-rq4-88",
    "hy3-oq2e-rq4-80",
    "hy3-oq2e-rq4-64",
    "hy3-oq2e-rq4-48",
    "hy3-oq2e-rq4-32",
)
KV_TOKEN_SWEEP = (4096, 16384, 32768, 65536)


@dataclass(frozen=True)
class _Apis:
    """Minimal stand-in for scripts/benchmark_q2_mtp_depth_matrix.RunnerAPIs.

    _runtime_config() only touches .config_factory and .parse_memory_bytes;
    everything else on the real RunnerAPIs (generate_ar, mx.reset_peak_memory,
    ...) requires `import mlx.core as mx` and is never referenced by the
    config-construction path this script exercises.
    """

    config_factory: Any
    parse_memory_bytes: Any


def build_config_and_request(
    preset: str, max_live_kv_tokens: int
) -> tuple[ExpertStreamingConfig, dict[str, Any]]:
    """Build the exact ExpertStreamingConfig + request dict that

    `scripts/benchmark_q2_mtp_depth_matrix.py --preset <preset>
    --max-live-kv-tokens <n>` would build, via the SAME preset-resolution and
    argparse machinery the real CLI uses (mtplx.benchmarks.presets +
    bqmdm.build_parser/_requests_from_args/_runtime_options_from_args/
    _runtime_config). Never calls apis.load().
    """

    parser = bqmdm.build_parser()
    argv = ["--preset", preset, "--max-live-kv-tokens", str(max_live_kv_tokens)]
    try:
        # environ={} keeps preset env exports (MTPLX_SUSTAINED_PREFILL etc.)
        # out of this process's real os.environ -- this script never runs
        # generation, so they are inert here regardless, but there is no
        # reason to mutate the caller's environment for a config-only probe.
        bqmdm.preselect_preset(parser, argv, environ={})
    except bqmdm.PresetError as exc:
        raise SystemExit(f"preset error resolving {preset!r}: {exc}") from exc
    args = parser.parse_args(argv)
    requests = bqmdm._requests_from_args(args)
    if len(requests) != 1:
        raise SystemExit(
            f"expected exactly one model request for preset {preset!r}, "
            f"got {len(requests)}"
        )
    # _requests_from_args() alone omits "model_key" (and other validated
    # fields); run_depth_matrix's real pipeline always passes each raw
    # request through _normalized_request() before use (see
    # _run_depth_matrix_impl -> normalized = [_normalized_request(request,
    # ...) for request in model_requests]). Do the same here so request["model_key"]
    # matches exactly what the real run would resolve.
    request = bqmdm._normalized_request(requests[0], mtp_disabled_baseline=False)
    # Mirror _run_depth_matrix_impl's exact layering (DEFAULT_RUNTIME_OPTIONS
    # < _runtime_options_from_args(args) < the separate top-level
    # trace_routes= parameter) -- _runtime_options_from_args() alone omits
    # "trace_routes", which _runtime_config() requires.
    options = {
        **bqmdm.DEFAULT_RUNTIME_OPTIONS,
        **bqmdm._runtime_options_from_args(args),
        "trace_routes": bool(args.trace_routes),
    }
    apis = _Apis(config_factory=ExpertStreamingConfig, parse_memory_bytes=parse_memory_bytes)
    config = bqmdm._runtime_config(apis, request["model_key"], options)
    return config, request


def evaluate_admission(
    preset: str,
    max_live_kv_tokens: int,
    *,
    memory_limit_override_bytes: int | None = None,
) -> dict[str, Any]:
    """Replicate mtplx/runtime.py's `_load_impl` config-time admission
    preflight for one (preset, kv_tokens) cell -- the CPU-only prefix up to
    and including the `plan.fits_fixed` check -- without ever calling
    apply_mlx_memory_cap / ExpertStreamingRuntime.open / apis.load().
    """

    config, request = build_config_and_request(preset, max_live_kv_tokens)
    if memory_limit_override_bytes is not None:
        replace_kwargs: dict[str, Any] = {
            "memory_limit_bytes": memory_limit_override_bytes
        }
        if config.island_layer_count is not None and config.island_layers:
            # ExpertStreamingConfig.__post_init__ eagerly resolves
            # island_layers from island_layer_count whenever the model spec
            # carries a static island_pin_order (true for hy3-expert-oq2e),
            # leaving BOTH fields populated on the constructed object --a
            # valid post-resolution state (memory_plan() only consults
            # len(island_layers)). dataclasses.replace() re-runs
            # __post_init__, whose entry-time mutual-exclusivity guard
            # would otherwise reject that same state it just produced.
            # island_layers alone is sufficient, so drop the now-redundant
            # count.
            replace_kwargs["island_layer_count"] = None
        config = dataclass_replace(config, **replace_kwargs)

    streaming_spec = get_model_spec(config.model_key)
    model_root = request["model_root"]
    model_config = load_config(model_root)

    mtp_precision = request.get("mtp_precision", "bf16")
    mtp_artifacts = request["mtp_artifacts"]
    # CPU-only: reads only the safetensors HEADER via os.pread (see
    # mtplx/hy3_mtp_patch.py:open_verified_hy3_mtp_artifacts) and returns
    # payload_bytes from the header's data_offsets. mx.load is never called
    # -- the runtime only calls it later, deep inside build_hy3_mtp_module,
    # which sits well after the fits_fixed gate this script stops at.
    with open_verified_hy3_mtp_artifacts(
        mtp_artifacts,
        precision=mtp_precision,
        expected_revision=streaming_spec.source_revision,
    ) as verified:
        streamed_mtp_resident_bytes = verified.payload_bytes

    hy3_router_incremental_bytes = 0
    if (
        str(model_config.get("model_type") or "") == "hy_v3"
        and config.hy3_router_kernel != "stock"
    ):
        # include_mtp=True: the depth-matrix lane always loads MTP (mtp=True)
        # for this schema; mirrors runtime.py's
        # `include_mtp=streamed_mtp_backend == "hy3" and bool(mtp)`.
        hy3_router_incremental_bytes = estimate_hy3_router_kernel_incremental_bytes(
            model_config, config.hy3_router_kernel, include_mtp=True
        )
    additional_resident_bytes = (
        streamed_mtp_resident_bytes + hy3_router_incremental_bytes
    )

    resolved_config = config
    if resolved_config.island_layer_count is not None:
        resolved_config = resolve_island_placement(resolved_config, model_root)

    resident_discount_bytes = 0
    if resolved_config.proj_quant or resolved_config.proj_requant:
        manifest = load_expert_manifest(request["manifest"])
        resident_discount_bytes = proj_quant_plan_discount(
            manifest, resolved_config.proj_quant
        ) + proj_requant_plan_discount(manifest, resolved_config.proj_requant)

    plan = resolved_config.memory_plan(
        streaming_spec,
        additional_resident_bytes=additional_resident_bytes,
        resident_discount_bytes=resident_discount_bytes,
    )

    result: dict[str, Any] = {
        "preset": preset,
        "max_live_kv_tokens": max_live_kv_tokens,
        "memory_limit_bytes": resolved_config.memory_limit_bytes,
        "memory_limit_override_applied": memory_limit_override_bytes is not None,
        "island_layer_count": len(resolved_config.island_layers),
        "streamed_mtp_resident_bytes": streamed_mtp_resident_bytes,
        "hy3_router_incremental_bytes": hy3_router_incremental_bytes,
        "additional_resident_bytes": additional_resident_bytes,
        "resident_discount_bytes": resident_discount_bytes,
        "fixed_bytes": plan.fixed_bytes,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "unallocated_bytes": plan.unallocated_bytes,
        "fits_fixed": plan.fits_fixed,
    }
    if plan.fits_fixed:
        result["status"] = "ADMIT"
        result["excess_bytes"] = 0
        result["implied_admission_total_bytes"] = (
            plan.fixed_bytes + plan.persistent_cache_bytes
        )
        result["implied_admission_total_gib"] = (
            plan.fixed_bytes + plan.persistent_cache_bytes
        ) / GIB
    else:
        result["status"] = "REJECT"
        result["excess_bytes"] = -plan.unallocated_bytes
        result["error_type"] = ExpertStreamingConfigurationError.__name__
        result["error_message"] = (
            "fixed expert-streaming footprint exceeds limit by "
            f"{-plan.unallocated_bytes} bytes"
        )
    return result


def sweep() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for preset in PRESETS:
        for kv_tokens in KV_TOKEN_SWEEP:
            base = evaluate_admission(preset, kv_tokens)
            cell: dict[str, Any] = {"preset": preset, "max_live_kv_tokens": kv_tokens, "base": base}
            if base["status"] == "REJECT":
                kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
                override_bytes = kv_delta_tokens * KV_BYTES_PER_TOKEN
                override_memory_limit_bytes = base["memory_limit_bytes"] + override_bytes
                overridden = evaluate_admission(
                    preset,
                    kv_tokens,
                    memory_limit_override_bytes=override_memory_limit_bytes,
                )
                cell["kv_delta_tokens"] = kv_delta_tokens
                cell["override_bytes"] = override_bytes
                cell["override_memory_limit_bytes"] = override_memory_limit_bytes
                cell["overridden"] = overridden
                cell["still_rejects_with_override"] = overridden["status"] == "REJECT"
            cells.append(cell)
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the sweep result as JSON to this path (default: stdout only).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override the ISO date stamped in the output (default: today, UTC).",
    )
    args = parser.parse_args()

    cells = sweep()
    payload = {
        "schema": "mtplx-envelope-admission-sweep-v1",
        "generated": (args.date or dt.date.today().isoformat()),
        "gate_exercised": {
            "file": "mtplx/runtime.py",
            "function": "_load_impl",
            "description": (
                "config-time admission preflight inside the streaming_requested "
                "branch: additional_resident_bytes (externally-resident bf16 MTP "
                "head payload_bytes + non-stock splitk router incremental bytes) "
                "is folded into config.memory_plan(...), then "
                "`if not streaming_plan.fits_fixed: raise "
                "ExpertStreamingConfigurationError(...)`. Same error class and "
                "identical message template as the internal duplicate gate in "
                "ExpertStreamingRuntime.open (mtplx/expert_runtime.py "
                "~line 2024-2027); the _load_impl gate is the one that actually "
                "fired for the 88x16k receipt (load_count=0)."
            ),
        },
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "presets": list(PRESETS),
        "kv_token_sweep": list(KV_TOKEN_SWEEP),
        "cells": cells,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
