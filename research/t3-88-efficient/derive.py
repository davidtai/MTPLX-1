#!/usr/bin/env python3
"""CPU-only admission-gate check for `hy3-oq2e-rq4-88e` (David's rule 5,
2026-07-22): drop the vestigial 1 GiB expert-cache-limit from
`hy3-oq2e-rq4-88r` (islands 79, full residency -- all 79 routed layers
resident, 0 streamed) before its context-matrix window.

CONFIG LEGALITY (read from source, not guessed):
  - mtplx/expert_runtime.py `ExpertStreamingConfig.__post_init__` validates
    `expert_cache_limit_bytes` via `_integer(name, value, minimum=0)` (lines
    223-230) -- an explicit 0 is LEGAL. This is unlike `island_layer_count`,
    which is validated with `minimum=1` (line 328) and rejects 0 outright.
    There is no analogous floor for the cache limit.
  - scripts/benchmark_q2_mtp_depth_matrix.py `_runtime_config` (lines
    2480-2484) special-cases the literal string "0" or "0b" to bypass
    `parse_memory_bytes` entirely and set `expert_cache_limit_bytes = 0`
    directly. `parse_memory_bytes` itself (mtplx/expert_runtime.py lines
    103-138) REJECTS any zero-valued magnitude for every other spelling
    (e.g. "0GiB" parses to `result = 0`, then `if result <= 0: raise
    ExpertStreamingConfigurationError("memory size must be positive")`).
    So the *only* TOML spelling that reaches a true zero is the bare
    string "0" (this preset uses exactly that).

BYTE-BUDGET FINDING (the reason this is a config-hygiene fix, not a
memory-reclaiming one): mtplx/expert_streaming_models.py `plan_expert_memory`
never folds `expert_cache_limit_bytes` into `fixed_bytes` (see lines
918-929) -- it only bounds `persistent_budget_bytes` (lines 933-934), which
feeds `persistent_cache_bytes`. At `hy3-oq2e-rq4-88r`'s islands=79 (==
`HY3_EXPERT_OQ2E.routed_layer_count`), `streamed_layer_count` is 0, so
`persistent_cache_bytes` is hardcoded to 0 (lines 948-951) UNCONDITIONALLY,
regardless of `expert_cache_limit_bytes`. Consequence: dropping the cache
limit from 1 GiB to 0 changes `fixed_bytes` / `unallocated_bytes` /
`fits_fixed` by exactly ZERO bytes at this island count -- there is no ~1 GiB
to "free" for the memory-limit knob. This script's own sweep (below)
confirms that empirically: `hy3-oq2e-rq4-88e`'s fixed_bytes at every KV
level is byte-identical to `hy3-oq2e-rq4-88r`'s (research/
envelope-admission-sweep-2026-07-22-88rf.json). memory-limit therefore STAYS
at 96 GiB, unchanged from -88r; -88 (95 GiB) is not reopened by this change.
The genuine benefit is config hygiene: the preset no longer carries a
declared cache byte-count that the runtime can structurally never spend
(0 loads measured at full residency), matching the full-residency preset's
own note ("--expert-cache-limit is vestigial here; it is inherited ... and
simply unused").

This script imports research/envelope_admission_sweep.py's own
`evaluate_admission()` / `KV_TOKEN_SWEEP` / `KV_BYTES_PER_TOKEN` /
`CONTROL_KV_TOKENS` directly (no re-implementation) -- same CPU-only
guarantee (no MLX allocation, no weight byte ever read).

Adds ONE thing the base harness does not sweep: kv4-priced admission (the
Lane B serving config). `kv_quant` is not a `--` flag on any -88 family
preset, so this script overrides it directly on the constructed
`ExpertStreamingConfig` via `dataclasses.replace` (identical technique to
`envelope_admission_sweep.evaluate_admission`'s own `memory_limit_override_
bytes` override) and re-plans. `HY3_EXPERT_OQ2E.kv_bytes_per_token_for`
locks q4 = 92,160 B/token (bf16 327,680 / 3.5556, matches
`AFFINE_QUANT_KEEP_NUMERATOR["q4"] = 9` -> 327_680 * 9 // 32 = 92_160)
against David's stated pricing exactly.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-88-efficient/derive.py \\
        [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
for _path in (REPO_ROOT, RESEARCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from envelope_admission_sweep import (  # noqa: E402
    CONTROL_KV_TOKENS,
    KV_BYTES_PER_TOKEN,
    KV_TOKEN_SWEEP,
    build_config_and_request,
    evaluate_admission,
)
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
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
PRESET = "hy3-oq2e-rq4-88e"
COMPARE_PRESET = "hy3-oq2e-rq4-88r"
KV4_CONTEXTS = (16384, 32768)


def sweep_cell(preset: str, kv_tokens: int) -> dict[str, Any]:
    """Same per-cell pattern as research/t3-88/preflight.py: evaluate at the
    preset's own declared memory-limit; only apply the KV-delta override if
    that REJECTs (isolates real KV growth from a baseline over-book)."""

    base = evaluate_admission(preset, kv_tokens)
    cell: dict[str, Any] = {"preset": preset, "max_live_kv_tokens": kv_tokens, "base": base}
    if base["status"] == "REJECT":
        kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
        override_bytes = kv_delta_tokens * KV_BYTES_PER_TOKEN
        override_memory_limit_bytes = base["memory_limit_bytes"] + override_bytes
        overridden = evaluate_admission(
            preset, kv_tokens, memory_limit_override_bytes=override_memory_limit_bytes
        )
        cell["kv_delta_tokens"] = kv_delta_tokens
        cell["override_bytes"] = override_bytes
        cell["override_memory_limit_bytes"] = override_memory_limit_bytes
        cell["overridden"] = overridden
        cell["still_rejects_with_override"] = overridden["status"] == "REJECT"
    return cell


def evaluate_admission_kv4(
    preset: str, max_live_kv_tokens: int, *, memory_limit_override_bytes: int | None = None
) -> dict[str, Any]:
    """Same CPU-only `_load_impl` preflight as envelope_admission_sweep's
    `evaluate_admission`, but with `kv_quant="q4"` overridden onto the
    constructed config via `dataclasses.replace` (kv4 is not a preset `--`
    flag on any -88-family entry, so this is the only way to price it
    without hand-deriving bytes). Mirrors evaluate_admission's own
    `memory_limit_override_bytes` technique exactly."""

    config, request = build_config_and_request(preset, max_live_kv_tokens)
    replace_kwargs: dict[str, Any] = {"kv_quant": "q4"}
    if memory_limit_override_bytes is not None:
        replace_kwargs["memory_limit_bytes"] = memory_limit_override_bytes
    if config.island_layer_count is not None and config.island_layers:
        # See envelope_admission_sweep.evaluate_admission's identical comment:
        # __post_init__ already resolved island_layers from island_layer_count
        # (true for hy3-expert-oq2e's static island_pin_order), so
        # dataclasses.replace() must drop the now-redundant count or its
        # mutual-exclusivity guard rejects the very state it produced.
        replace_kwargs["island_layer_count"] = None
    config = dataclass_replace(config, **replace_kwargs)

    streaming_spec = get_model_spec(config.model_key)
    model_root = request["model_root"]
    from mtplx.artifacts import load_config

    model_config = load_config(model_root)

    mtp_precision = request.get("mtp_precision", "bf16")
    mtp_artifacts = request["mtp_artifacts"]
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
        hy3_router_incremental_bytes = estimate_hy3_router_kernel_incremental_bytes(
            model_config, config.hy3_router_kernel, include_mtp=True
        )
    additional_resident_bytes = streamed_mtp_resident_bytes + hy3_router_incremental_bytes

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
        "kv_quant": "q4",
        "kv_bytes_per_token": streaming_spec.kv_bytes_per_token_for("q4"),
        "memory_limit_bytes": resolved_config.memory_limit_bytes,
        "memory_limit_override_applied": memory_limit_override_bytes is not None,
        "island_layer_count": len(resolved_config.island_layers),
        "fixed_bytes": plan.fixed_bytes,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "unallocated_bytes": plan.unallocated_bytes,
        "fits_fixed": plan.fits_fixed,
        "status": "ADMIT" if plan.fits_fixed else "REJECT",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # 1) Full bf16-pricing sweep at the real KV_TOKEN_SWEEP, same shape as
    #    research/t3-88/preflight.py, for both -88e and its parent -88r (to
    #    show byte-identical fixed_bytes -- the zero-effect finding).
    bf16_cells = {
        preset: [sweep_cell(preset, kv) for kv in KV_TOKEN_SWEEP]
        for preset in (PRESET, COMPARE_PRESET)
    }
    identical = all(
        e["base"]["fixed_bytes"] == r["base"]["fixed_bytes"]
        for e, r in zip(bf16_cells[PRESET], bf16_cells[COMPARE_PRESET])
    )

    # 2) Does dropping the cache re-open the original -88 95 GiB limit? Test
    #    -88e's own fixed_bytes against a 95 GiB override at the control KV.
    limit_95_bytes = 95 * GIB
    limit_95_check = evaluate_admission(
        PRESET, CONTROL_KV_TOKENS, memory_limit_override_bytes=limit_95_bytes
    )

    # 3) kv4-priced admission at the mandatory real-prefill contexts.
    kv4_cells = [evaluate_admission_kv4(PRESET, kv) for kv in KV4_CONTEXTS]

    for preset, cells in bf16_cells.items():
        for cell in cells:
            final = cell.get("overridden", cell["base"])
            print(
                f"{preset:24} kv={cell['max_live_kv_tokens']:6} bf16   "
                f"{final['status']:6} fixed={final['fixed_bytes'] / GIB:9.4f} GiB "
                f"limit={final['memory_limit_bytes'] / GIB:9.4f} GiB "
                f"margin={final['unallocated_bytes'] / GIB:+8.4f} GiB",
                file=sys.stderr,
            )
    print(
        f"fixed_bytes identical -88e vs -88r across {KV_TOKEN_SWEEP}: {identical}",
        file=sys.stderr,
    )
    print(
        f"{PRESET:24} kv={CONTROL_KV_TOKENS:6} bf16   "
        f"@95GiB override: {limit_95_check['status']:6} "
        f"fixed={limit_95_check['fixed_bytes'] / GIB:9.4f} GiB "
        f"margin={limit_95_check['unallocated_bytes'] / GIB:+8.4f} GiB",
        file=sys.stderr,
    )
    for cell in kv4_cells:
        print(
            f"{PRESET:24} kv={cell['max_live_kv_tokens']:6} kv4    "
            f"{cell['status']:6} fixed={cell['fixed_bytes'] / GIB:9.4f} GiB "
            f"limit={cell['memory_limit_bytes'] / GIB:9.4f} GiB "
            f"margin={cell['unallocated_bytes'] / GIB:+8.4f} GiB",
            file=sys.stderr,
        )

    payload = {
        "schema": "mtplx-t3-88-efficient-v1",
        "generated": "2026-07-22",
        "note": (
            "David's rule 5 CPU work: drop the vestigial 1 GiB expert-cache-"
            "limit from hy3-oq2e-rq4-88r (islands 79, full residency) -> "
            "hy3-oq2e-rq4-88e (expert-cache-limit '0'). Confirms (a) config "
            "legality of an explicit zero cache limit, (b) fixed_bytes is "
            "BYTE-IDENTICAL to -88r at every swept KV level (expert-cache-"
            "limit never enters fixed_bytes when streamed_layer_count==0), "
            "so memory-limit stays 96 GiB (the original -88 95 GiB limit is "
            "NOT reopened), and (c) kv4-priced admission at the mandatory "
            "real-prefill contexts."
        ),
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "kv_bytes_per_token_bf16": KV_BYTES_PER_TOKEN,
        "kv_token_sweep": list(KV_TOKEN_SWEEP),
        "bf16_cells": bf16_cells,
        "fixed_bytes_identical_to_88r": identical,
        "limit_95_reopen_check": limit_95_check,
        "kv4_contexts": list(KV4_CONTEXTS),
        "kv4_cells": kv4_cells,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")

    ok = (
        identical
        and limit_95_check["status"] == "REJECT"
        and all(
            cell.get("overridden", cell["base"])["status"] == "ADMIT"
            for cell in bf16_cells[PRESET]
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
