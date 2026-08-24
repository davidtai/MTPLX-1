#!/usr/bin/env python3
"""Matched 16K ABBA gate for replacing fixed-D3 MTP with DFlash2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen38_challenge_port_gate import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_LOCK,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    _context_prompt_to_token_count,
    _load_optimized_speed_stack,
    _projection_counter_snapshot,
    _counter_delta,
    _read_prompt,
    _route_execution_options,
    _run_arm,
)


DFLASH_REPO = "z-lab/Qwen3.8-27B-DFlash2"
DFLASH_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DFLASH_SOURCE_COMMIT = "c5b76ddb62bdefb6eeef1282641842edcf23a1b8"
PROMOTION_THRESHOLD_PCT = 0.05
STATIC_WIDTH = 8
DFLASH_SURVIVOR_ROWS = frozenset({21, 24, 26, 48})
DFLASH_ADAPTIVE_ROWS = (11, 15, 18, 24, 25, 26, 32)
DFLASH_CUSTOM_ROWS = (34, 40, 47)
DFLASH_CUSTOM_WIDTHS = {34: (6,), 40: (6, 7), 47: (6, 7, 8)}
DFLASH_GQA_WIDTHS = (6, 7, 8)
FULL_RETAINED_ROUTE = (
    "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
    "r20_kv_only_history+r21_qk_rms_rope+r24_eval_ladder+"
    "r26_prefill_ladder_3+r36_qkv_islands+r48_boundary_fused+"
    "r50_wired_residency+r53_command_buffers+r61_dual_norm_concat"
)
DEFAULT_DFLASH_SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/"
    f"snapshots/{DFLASH_REVISION}"
)
DEFAULT_ROW36_ARTIFACT = Path.home() / (
    ".cache/huggingface/hub/"
    "models--amal-david--qwen38-mtp-head-q4-qkv-islands-v1/"
    "blobs/517bb133d7ca6e228a5129710b3cb2c25aa9944753b9f9a225fa1e8135df5e65"
)


def _token_hash(tokens: list[int] | tuple[int, ...]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _dflash_target_sampling(*, seed: int) -> Iterator[None]:
    """Install the campaign's exact target sampler at DFlash's posterior seam.

    DFlash2 still proposes greedily.  Each verified target row is sampled from
    temperature 1 / top-p .95 / top-k 20, and drafted tokens are accepted only
    while they match those target samples.  On the first mismatch the sampled
    target token is emitted, preserving the target sampler distribution.
    """

    import mlx.core as mx

    from dflash_mlx.engine import spec_epoch
    from mtplx.fast_sampling import sample_token_ids_from_mlx_logits
    from mtplx.sampling import SamplerConfig

    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    original = spec_epoch.greedy_tokens_with_mask

    def sample_target_rows(logits, suppress_token_mask=None):
        if suppress_token_mask is not None:
            raise RuntimeError(
                "Qwen 3.8 DFlash benchmark does not permit token suppression"
            )
        sampled = sample_token_ids_from_mlx_logits(logits, sampler)
        if sampled is None:
            raise RuntimeError("DFlash target sampler could not stay on device")
        return sampled.astype(mx.uint32)

    mx.random.seed(int(seed))
    spec_epoch.greedy_tokens_with_mask = sample_target_rows
    try:
        yield
    finally:
        spec_epoch.greedy_tokens_with_mask = original


def _install_retained_route(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    *,
    row36_artifact: Path,
) -> Any:
    from mtplx.qwen38_challenge import install_qwen38_route

    options = _route_execution_options(FULL_RETAINED_ROUTE)
    return install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route=str(options["cache_route"]),
        dual_norm=bool(options["dual_norm"]),
        source_proposal=False,
        row10_compact_vocab=bool(options["row10_compact_vocab"]),
        mtp_block_variant=options["mtp_block_variant"],
        mtp_block_artifact_path=row36_artifact,
        row18_gdn_decay_memo=bool(options["row18_gdn_decay_memo"]),
        row21_qk_rms_rope=bool(options["row21_qk_rms_rope"]),
        row24_eval_ladder=bool(options["row24_eval_ladder"]),
        row26_prefill_ladder_3=bool(options["row26_prefill_ladder_3"]),
        row48_boundary_fused=bool(options["row48_boundary_fused"]),
        row50_wired_residency=bool(options["row50_wired_residency"]),
        row63_q8_embedding_dual_norm=False,
        row70_qmv_sumtable=False,
        row78_qmv_active_groups=False,
        row80_qmv_m2=False,
    )


def _parse_dflash_survivors(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = set(rows) - DFLASH_SURVIVOR_ROWS
    if unknown:
        raise ValueError(f"unsupported DFlash survivor rows: {sorted(unknown)}")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError("DFlash survivor rows must be unique and chronological")
    dependencies = {21: (), 24: (21,), 26: (21, 24), 48: ()}
    for row, required in dependencies.items():
        if row in rows and not set(required) <= set(rows):
            raise ValueError(f"DFlash row {row} requires survivor rows {required}")
    return rows


def _parse_dflash_adaptive_rows(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = set(rows) - set(DFLASH_ADAPTIVE_ROWS)
    if unknown:
        raise ValueError(f"unsupported DFlash adaptive rows: {sorted(unknown)}")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError("DFlash adaptive rows must be unique and chronological")
    if rows[0] != 11:
        raise ValueError("DFlash adaptive proposal stack requires row 11")
    return rows


def _parse_dflash_custom_rows(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = set(rows) - set(DFLASH_CUSTOM_ROWS)
    if unknown:
        raise ValueError(f"unsupported DFlash custom rows: {sorted(unknown)}")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError("DFlash custom rows must be unique and chronological")
    expected_prefix = DFLASH_CUSTOM_ROWS[: len(rows)]
    if rows != expected_prefix:
        raise ValueError("DFlash custom rows must be dependency-closed")
    return rows


def _parse_dflash_gqa_widths(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    widths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if widths != DFLASH_GQA_WIDTHS:
        raise ValueError("DFlash GQA widths must be empty or exactly 6,7,8")
    return widths


def _install_dflash_route(
    runtime: Any,
    *,
    survivor_rows: tuple[int, ...],
    gqa_widths: tuple[int, ...] = (),
    m8_nax_island: bool = False,
    m8_nax_output: bool = True,
    m8_linear_z: bool = False,
    m7_nax_output: bool = False,
    m7_nax_linear_z: bool = False,
    m8_nax_expanded: bool = False,
    m8_nax_kv: bool = False,
    m8_nax_qkv: bool = False,
    m8_nax_mlp: bool = False,
    m5_exact: bool = False,
    m6_kp1: bool = False,
    row24_prefill_ladder: bool = True,
    row24_decode_ladder: bool = True,
    row48_prefill_fused: bool = True,
    row48_decode_fused: bool = True,
) -> Any:
    """Install only survivor mechanisms that remain valid on DFlash target work."""

    from mtplx.gdn_capture import configure_qwen38_dflash_row48_boundary
    from mtplx.qwen38_challenge_kernels import (
        configure_qwen38_dflash_row24_eval_ladder,
        configure_qwen38_dflash_gqa_widths,
        configure_qwen38_dflash_m8_nax_island,
        configure_qwen38_row21_qk_rms_rope,
        configure_qwen38_row24_qk_length_limit,
    )

    rows = set(survivor_rows)
    row21_report = configure_qwen38_row21_qk_rms_rope(
        runtime.model,
        active=21 in rows,
    )
    row24_report = configure_qwen38_row24_qk_length_limit(
        runtime.model,
        active=24 in rows,
        max_length=32 if 26 in rows else 16,
    )
    row24_ladder_report = configure_qwen38_dflash_row24_eval_ladder(
        runtime.model,
        active=24 in rows,
        prefill_stride=3 if 26 in rows else 4,
        prefill_active=bool(row24_prefill_ladder),
        decode_active=bool(row24_decode_ladder),
    )
    row48_report = configure_qwen38_dflash_row48_boundary(
        runtime.model,
        active=48 in rows,
        prefill_active=bool(row48_prefill_fused),
        decode_active=bool(row48_decode_fused),
    )
    gqa_report = configure_qwen38_dflash_gqa_widths(
        runtime.model,
        active=bool(gqa_widths),
        widths=DFLASH_GQA_WIDTHS,
    )
    m8_nax_report = configure_qwen38_dflash_m8_nax_island(
        runtime.model,
        active=bool(m8_nax_island),
        include_m8_output=bool(m8_nax_output),
        include_linear_z=bool(m8_linear_z),
        include_m7_output=bool(m7_nax_output),
        include_m7_linear_z=bool(m7_nax_linear_z),
        include_m8_expanded=bool(m8_nax_expanded),
        include_m8_kv=bool(m8_nax_kv),
        include_m8_qkv=bool(m8_nax_qkv),
        include_m8_mlp=bool(m8_nax_mlp),
        include_m5_exact=bool(m5_exact),
        include_m6_kp1=bool(m6_kp1),
    )
    from mtplx.nax_verify import configure_qwen38_nax_split_tuning

    nax_split_report = configure_qwen38_nax_split_tuning(
        active=bool(m8_nax_island),
        m7_linear_z_nsg4=os.environ.get(
            "MTPLX_QWEN38_M7_LINEAR_Z_NSG4", "0"
        ) == "1",
        m8_kv_nsg16=os.environ.get("MTPLX_QWEN38_M8_KV_NSG16", "0") == "1",
        m8_qkv_nsg4=os.environ.get("MTPLX_QWEN38_M8_QKV_NSG4", "0") == "1",
    )
    feature_receipt: dict[str, dict[str, Any]] = {}
    if 21 in rows:
        feature_receipt["r21_qk_rms_rope"] = row21_report
    if 24 in rows:
        feature_receipt["r24_qk_length_limit"] = row24_report
        feature_receipt["r24_eval_ladder"] = row24_ladder_report
    if 26 in rows:
        feature_receipt["r26_prefill_ladder_3"] = {"active": 1}
    if 48 in rows:
        feature_receipt["r48_boundary_fused"] = row48_report
    if gqa_widths:
        feature_receipt["dflash_gqa_widths"] = gqa_report
    if m8_nax_island:
        feature_receipt["dflash_m8_nax_island"] = m8_nax_report
        feature_receipt["dflash_nax_split_tuning"] = nax_split_report
    runtime.qwen38_feature_receipt = feature_receipt
    return SimpleNamespace(
        route_id="+".join(
            (
                "dflash2_static8",
                *(f"r{row:02d}" for row in survivor_rows),
                *(("gqa678",) if gqa_widths else ()),
                *(("m8nax",) if m8_nax_island else ()),
                *(("m8noout",) if m8_nax_island and not m8_nax_output else ()),
                *(("m8linearz",) if m8_linear_z else ()),
                *(("m7naxout",) if m7_nax_output else ()),
                *(("m7naxlinearz",) if m7_nax_linear_z else ()),
                *(("m8naxexpanded",) if m8_nax_expanded else ()),
                *(("m8naxkv",) if m8_nax_kv else ()),
                *(("m8naxqkv",) if m8_nax_qkv else ()),
                *(("m8naxmlp",) if m8_nax_mlp else ()),
                *(("m5exact",) if m5_exact else ()),
                *(("m6kp1",) if m6_kp1 else ()),
                *(("r24noprefill",) if not row24_prefill_ladder else ()),
                *(("r24nodecode",) if not row24_decode_ladder else ()),
                *(("r48noprefill",) if not row48_prefill_fused else ()),
                *(("r48nodecode",) if not row48_decode_fused else ()),
            )
        )
    )


def _load_optimized_speed_target_stack(
    model_path: Path,
    runtime_contract: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply the Optimized-Speed profile while never constructing native MTP."""

    from mtplx.runtime import load

    def load_target_only(path: Path, *, mtp: bool) -> Any:
        if not mtp:
            raise AssertionError("Optimized-Speed loader contract changed")
        return load(path, mtp=False)

    def skip_native_draft_head(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"installed": False, "reason": "replaced_by_dflash2"}

    runtime, stack = _load_optimized_speed_stack(
        model_path,
        runtime_contract,
        load_runtime_fn=load_target_only,
        install_draft_head_fn=skip_native_draft_head,
    )
    return runtime, {**stack, "native_mtp_loaded": False}


def _dflash_target_counter_snapshot() -> dict[str, int]:
    return {}


def _flat_counter_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _run_dflash_arm(
    bundle: Any,
    prompt_ids: list[int],
    runtime_context: Any,
    *,
    max_tokens: int,
    seed: int,
    route: Any,
    survivor_rows: tuple[int, ...],
    adaptive_rows: tuple[int, ...],
    custom_rows: tuple[int, ...],
    gqa_widths: tuple[int, ...],
    m8_nax_island: bool,
    m8_nax_output: bool,
    m8_linear_z: bool,
    m7_nax_output: bool,
    m7_nax_linear_z: bool,
    m8_nax_expanded: bool,
    m8_nax_kv: bool,
    m8_nax_qkv: bool,
    m8_nax_mlp: bool,
    m5_exact: bool,
    m6_kp1: bool,
    row24_prefill_ladder: bool,
    row24_decode_ladder: bool,
    row48_prefill_fused: bool,
    row48_decode_fused: bool,
    cost_aligned_widths: bool,
    release_report: dict[str, bool],
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.benchmarks.runners.dflash2_depth_sweep import (
        run_dflash2_candidate,
    )

    counters_before = _projection_counter_snapshot()
    dflash_counters_before = _dflash_target_counter_snapshot()
    mx.reset_peak_memory()
    started = time.perf_counter()
    with _dflash_target_sampling(seed=seed):
        arm = run_dflash2_candidate(
            bundle,
            prompt_ids,
            STATIC_WIDTH,
            runtime_context,
            max_tokens=max_tokens,
        )
    wall_s = time.perf_counter() - started
    counters_after = _projection_counter_snapshot()
    dflash_counters_after = _dflash_target_counter_snapshot()
    tokens = tuple(int(token) for token in arm.pop("tokens"))
    return {
        **arm,
        "engine": "dflash2",
        "route_id": "+".join(
            (
                "dflash2_static8",
                *(f"r{row:02d}" for row in survivor_rows),
                *(f"a{row:02d}" for row in adaptive_rows),
                *(f"c{row:02d}" for row in custom_rows),
                *(("gqa678",) if gqa_widths else ()),
                *(("m8nax",) if m8_nax_island else ()),
                *(("m8noout",) if m8_nax_island and not m8_nax_output else ()),
                *(("m8linearz",) if m8_linear_z else ()),
                *(("m7naxout",) if m7_nax_output else ()),
                *(("m7naxlinearz",) if m7_nax_linear_z else ()),
                *(("m8naxexpanded",) if m8_nax_expanded else ()),
                *(("m8naxkv",) if m8_nax_kv else ()),
                *(("m8naxqkv",) if m8_nax_qkv else ()),
                *(("m8naxmlp",) if m8_nax_mlp else ()),
                *(("m5exact",) if m5_exact else ()),
                *(("m6kp1",) if m6_kp1 else ()),
                *(("r24noprefill",) if not row24_prefill_ladder else ()),
                *(("r24nodecode",) if not row24_decode_ladder else ()),
                *(("r48noprefill",) if not row48_prefill_fused else ()),
                *(("r48nodecode",) if not row48_decode_fused else ()),
                *(("cost_aligned",) if cost_aligned_widths else ()),
            )
        ),
        "installed_route_id": route.route_id,
        "wall_s": wall_s,
        "token_hash": _token_hash(tokens),
        "tokens": list(tokens),
        "engagement": _counter_delta(counters_before, counters_after),
        "dflash_target_engagement": _flat_counter_delta(
            dflash_counters_before,
            dflash_counters_after,
        ),
        "feature_receipt": dict(
            getattr(bundle.runtime, "qwen38_feature_receipt", {}) or {}
        )
        | {
            "native_mtp_release": dict(release_report),
            "r53_command_buffers": {
                "max_mb_per_buffer": int(os.environ["MLX_MAX_MB_PER_BUFFER"]),
                "max_ops_per_buffer": int(os.environ["MLX_MAX_OPS_PER_BUFFER"]),
            },
        },
    }


def _run_mtp_arm(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    seed: int,
    row36_artifact: Path,
) -> dict[str, Any]:
    arm = _run_arm(
        runtime,
        config,
        model_path,
        prompt_ids,
        route_id=FULL_RETAINED_ROUTE,
        max_tokens=max_tokens,
        seed=seed,
        target_temperature=1.0,
        draft_temperature=1.0,
        source_artifact_path=None,
        row17_artifact_path=None,
        row28_artifact_path=None,
        row36_artifact_path=row36_artifact,
    )
    arm["engine"] = "mtp_fixed_d3"
    arm["prefill_tps"] = float(arm["prefill_tok_s"])
    arm["decode_tps"] = float(arm["decode_tok_s"])
    arm["peak_memory_gb"] = float(arm["peak_memory_gib"])
    return arm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DFLASH_SNAPSHOT)
    parser.add_argument("--row36-artifact", type=Path, default=DEFAULT_ROW36_ARTIFACT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--release-native-mtp", action="store_true")
    parser.add_argument("--dflash-survivors", default="")
    parser.add_argument("--dflash-adaptive-rows", default="")
    parser.add_argument("--dflash-custom-rows", default="")
    parser.add_argument("--dflash-gqa-widths", default="")
    parser.add_argument("--dflash-m8-nax-island", action="store_true")
    parser.add_argument("--disable-dflash-m8-output", action="store_true")
    parser.add_argument("--dflash-m8-linear-z", action="store_true")
    parser.add_argument("--dflash-m7-nax-output", action="store_true")
    parser.add_argument("--dflash-m7-nax-linear-z", action="store_true")
    parser.add_argument("--dflash-m8-nax-expanded", action="store_true")
    parser.add_argument("--dflash-m8-nax-kv", action="store_true")
    parser.add_argument("--dflash-m8-nax-qkv", action="store_true")
    parser.add_argument("--dflash-m8-nax-mlp", action="store_true")
    parser.add_argument("--dflash-m5-exact", action="store_true")
    parser.add_argument("--dflash-m6-kp1", action="store_true")
    parser.add_argument("--disable-dflash-row24-prefill-ladder", action="store_true")
    parser.add_argument("--disable-dflash-row24-decode-ladder", action="store_true")
    parser.add_argument("--disable-dflash-row48-prefill-fusion", action="store_true")
    parser.add_argument("--disable-dflash-row48-decode-fusion", action="store_true")
    parser.add_argument("--dflash-cost-aligned-widths", action="store_true")
    parser.add_argument(
        "--engine",
        choices=("mtp_fixed_d3", "dflash2"),
        required=True,
        help="Run one isolated conditioner plus one timed arm.",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prompt_tokens != 16_384 or args.max_tokens != 1024:
        raise ValueError("item 55 requires exactly 16K input and 1024 output tokens")
    model_path = args.model.expanduser().resolve()
    draft_path = args.draft.expanduser().resolve()
    row36_artifact = args.row36_artifact.expanduser().resolve()
    if not draft_path.is_dir():
        raise FileNotFoundError(f"pinned DFlash snapshot is absent: {draft_path}")
    if args.engine == "mtp_fixed_d3" and not row36_artifact.is_file():
        raise FileNotFoundError(f"row 36 artifact is absent: {row36_artifact}")
    survivor_rows = _parse_dflash_survivors(args.dflash_survivors)
    adaptive_rows = _parse_dflash_adaptive_rows(args.dflash_adaptive_rows)
    custom_rows = _parse_dflash_custom_rows(args.dflash_custom_rows)
    gqa_widths = _parse_dflash_gqa_widths(args.dflash_gqa_widths)
    if args.dflash_m8_linear_z and not args.dflash_m8_nax_island:
        raise ValueError("DFlash M8 linear-Z requires the M8 NAX island")
    if args.dflash_m7_nax_output and not args.dflash_m8_nax_island:
        raise ValueError("DFlash M7 output route requires the M8 NAX island")
    if args.dflash_m7_nax_linear_z and not args.dflash_m7_nax_output:
        raise ValueError("DFlash M7 linear-Z route requires the M7 output route")
    if args.dflash_m8_nax_expanded and not args.dflash_m8_nax_island:
        raise ValueError("DFlash expanded M8 route requires the M8 NAX island")
    if args.dflash_m8_nax_kv and not args.dflash_m8_nax_island:
        raise ValueError("DFlash M8 K/V route requires the M8 NAX island")
    if args.disable_dflash_m8_output and not args.dflash_m8_nax_island:
        raise ValueError("Disabling DFlash M8 output requires the NAX verify route")
    if (args.dflash_m5_exact or args.dflash_m6_kp1) and not args.dflash_m8_nax_island:
        raise ValueError("DFlash M5/M6 tuning requires the NAX verify route")
    if args.dflash_m6_kp1 and not args.dflash_m5_exact:
        raise ValueError("DFlash M6 tuning requires the retained exact-M5 route")

    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    guarded_by_parent = _verify_parent_guard_attestation(args.lock)
    lock_handle = None
    if not guarded_by_parent:
        lock_handle = args.lock.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {args.lock}") from exc

    # The retained Qwen3.8 stack is the turbo profile.  Establish its complete
    # process-latched environment before runtime loading so isolated ABBA arms
    # cannot silently benchmark the stock/default verify path.
    from mtplx.profiles import apply_profile_env

    apply_profile_env("turbo")
    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    from mtplx.backends.registry import load_runtime_contract

    runtime_contract, contract_error = load_runtime_contract(model_path)
    if contract_error is not None:
        raise RuntimeError(f"invalid runtime contract: {contract_error}")
    raw_contract = {} if runtime_contract is None else runtime_contract.raw
    if args.engine == "dflash2" and args.release_native_mtp:
        runtime, optimized_stack = _load_optimized_speed_target_stack(
            model_path,
            raw_contract,
        )
    else:
        runtime, optimized_stack = _load_optimized_speed_stack(
            model_path,
            raw_contract,
        )

    from mtplx.artifacts import load_config

    config = load_config(model_path)
    bundle = None
    runtime_context = None
    dflash_route = None
    release_report = {
        "native_mtp_released": bool(args.release_native_mtp),
        "native_mtp_loaded": not bool(args.release_native_mtp),
    }
    if args.engine == "dflash2":
        from mtplx.benchmarks.dflash2_runtime import bind_mtplx_dflash2_bundle
        from mtplx.benchmarks.runners.dflash2_depth_sweep import (
            build_fixed_dflash_runtime_context,
        )

        bundle = bind_mtplx_dflash2_bundle(runtime, str(draft_path))
        dflash_route = _install_dflash_route(
            runtime,
            survivor_rows=survivor_rows,
            gqa_widths=gqa_widths,
            m8_nax_island=bool(args.dflash_m8_nax_island),
            m8_nax_output=not bool(args.disable_dflash_m8_output),
            m8_linear_z=bool(args.dflash_m8_linear_z),
            m7_nax_output=bool(args.dflash_m7_nax_output),
            m7_nax_linear_z=bool(args.dflash_m7_nax_linear_z),
            m8_nax_expanded=bool(args.dflash_m8_nax_expanded),
            m8_nax_kv=bool(args.dflash_m8_nax_kv),
            m8_nax_qkv=bool(args.dflash_m8_nax_qkv),
            m8_nax_mlp=bool(args.dflash_m8_nax_mlp),
            m5_exact=bool(args.dflash_m5_exact),
            m6_kp1=bool(args.dflash_m6_kp1),
            row24_prefill_ladder=not bool(args.disable_dflash_row24_prefill_ladder),
            row24_decode_ladder=not bool(args.disable_dflash_row24_decode_ladder),
            row48_prefill_fused=not bool(args.disable_dflash_row48_prefill_fusion),
            row48_decode_fused=not bool(args.disable_dflash_row48_decode_fusion),
        )
        if dflash_route is None:
            raise RuntimeError("DFlash2 survivor route did not install")
        from mtplx.qwen38_dflash_adaptive import (
            configure_qwen38_dflash_adaptive_policy,
        )

        adaptive_report = configure_qwen38_dflash_adaptive_policy(
            bundle.target_model,
            active=bool(adaptive_rows),
            proposal_rows=adaptive_rows,
            cost_aligned_widths=bool(args.dflash_cost_aligned_widths),
        )
        from mtplx.qwen38_qmv import configure_qwen38_dflash_qmv

        custom_report = configure_qwen38_dflash_qmv(
            bundle.draft_model,
            active=bool(custom_rows),
            allowed_widths=(
                DFLASH_CUSTOM_WIDTHS[custom_rows[-1]] if custom_rows else ()
            ),
        )
        from mtplx.qwen38_challenge import configure_qwen38_row50_wired_residency

        row50_report = configure_qwen38_row50_wired_residency(runtime, active=True)
        if not bool(row50_report.get("installed")):
            raise RuntimeError("DFlash2 row-50 wired residency did not install")
        runtime.qwen38_feature_receipt = {
            **dict(getattr(runtime, "qwen38_feature_receipt", {}) or {}),
            "adaptive_policy": adaptive_report,
            "custom_draft_qmv": custom_report,
            "r50_wired_residency": row50_report,
        }
        runtime_context = build_fixed_dflash_runtime_context()
    prompt_id, instruction = _read_prompt(args.prompt_file)
    prompt_text, prompt_ids = _context_prompt_to_token_count(
        runtime.tokenizer,
        context=args.context_file.read_text(encoding="utf-8"),
        instruction=instruction,
        target_tokens=args.prompt_tokens,
    )
    del prompt_text

    def run(tokens: int) -> dict[str, Any]:
        if args.engine == "mtp_fixed_d3":
            return _run_mtp_arm(
                runtime,
                config,
                model_path,
                prompt_ids,
                max_tokens=tokens,
                seed=args.seed,
                row36_artifact=row36_artifact,
            )
        if bundle is None or runtime_context is None or dflash_route is None:
            raise RuntimeError("DFlash2 isolated child did not construct its bundle")
        return _run_dflash_arm(
            bundle,
            prompt_ids,
            runtime_context,
            max_tokens=tokens,
            seed=args.seed,
            route=dflash_route,
            survivor_rows=survivor_rows,
            adaptive_rows=adaptive_rows,
            custom_rows=custom_rows,
            gqa_widths=gqa_widths,
            m8_nax_island=bool(args.dflash_m8_nax_island),
            m8_nax_output=not bool(args.disable_dflash_m8_output),
            m8_linear_z=bool(args.dflash_m8_linear_z),
            m7_nax_output=bool(args.dflash_m7_nax_output),
            m7_nax_linear_z=bool(args.dflash_m7_nax_linear_z),
            m8_nax_expanded=bool(args.dflash_m8_nax_expanded),
            m8_nax_kv=bool(args.dflash_m8_nax_kv),
            m8_nax_qkv=bool(args.dflash_m8_nax_qkv),
            m8_nax_mlp=bool(args.dflash_m8_nax_mlp),
            m5_exact=bool(args.dflash_m5_exact),
            m6_kp1=bool(args.dflash_m6_kp1),
            row24_prefill_ladder=not bool(args.disable_dflash_row24_prefill_ladder),
            row24_decode_ladder=not bool(args.disable_dflash_row24_decode_ladder),
            row48_prefill_fused=not bool(args.disable_dflash_row48_prefill_fusion),
            row48_decode_fused=not bool(args.disable_dflash_row48_decode_fusion),
            cost_aligned_widths=bool(args.dflash_cost_aligned_widths),
            release_report=release_report,
        )

    warmup = run(args.warmup_tokens)
    arm = run(args.max_tokens)
    receipt = {
        "kind": "qwen38_challenge_dflash2_item55_isolated_arm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": args.engine,
        "model": str(model_path),
        "dflash": {
            "repo_id": DFLASH_REPO,
            "revision": DFLASH_REVISION,
            "snapshot": str(draft_path),
            "source_commit": DFLASH_SOURCE_COMMIT,
            "block_size": 8,
            "static_width": STATIC_WIDTH,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
        "workload": {
            "prompt_id": prompt_id,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": _token_hash(prompt_ids),
            "generated_tokens": args.max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": args.seed,
            "conditioning_tokens_per_engine": args.warmup_tokens,
        },
        "optimized_speed_stack": optimized_stack,
        "retained_route": FULL_RETAINED_ROUTE,
        "dflash_survivor_rows": list(survivor_rows),
        "dflash_adaptive_rows": list(adaptive_rows),
        "dflash_custom_rows": list(custom_rows),
        "dflash_gqa_widths": list(gqa_widths),
        "dflash_m8_nax_island": bool(args.dflash_m8_nax_island),
        "dflash_m8_linear_z": bool(args.dflash_m8_linear_z),
        "dflash_cost_aligned_widths": bool(args.dflash_cost_aligned_widths),
        "native_mtp_release": dict(release_report),
        "mlx_version": importlib.metadata.version("mlx"),
        "dflash_mlx_version": importlib.metadata.version("dflash-mlx"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "warmup": warmup,
        "arm": arm,
        "gpu_lock_scope": str(args.lock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "engine": args.engine,
                "prefill_tps": arm["prefill_tps"],
                "decode_tps": arm["decode_tps"],
                "peak_memory_gb": arm["peak_memory_gb"],
                "wall_s": arm["wall_s"],
            },
            sort_keys=True,
        )
    )
    if lock_handle is not None:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
