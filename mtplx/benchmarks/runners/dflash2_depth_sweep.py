"""Greedy Qwen3.8 DFlash2 depth-sweep adapters and orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
import hashlib
import math
from typing import Any

from mtplx.benchmarks.dflash2_contract import DepthBracket, select_stock_depth
from mtplx.sampling import SamplerConfig


GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
MTP_DEPTH = 3


def _generate_ar(*args, **kwargs):
    from mtplx.generation import generate_ar

    return generate_ar(*args, **kwargs)


def _generate_mtpk(*args, **kwargs):
    from mtplx.generation import generate_mtpk

    return generate_mtpk(*args, **kwargs)


def _build_offline_runtime_context(**kwargs):
    from dflash_mlx.runtime.context import build_offline_runtime_context

    return build_offline_runtime_context(**kwargs)


def _stream_dflash_generate(**kwargs):
    from dflash_mlx.runtime import stream_dflash_generate

    return stream_dflash_generate(**kwargs)


def _exact_tokens(tokens: Iterable[int], *, expected_tokens: int, engine: str) -> tuple[int, ...]:
    token_ids = tuple(int(token) for token in tokens)
    if len(token_ids) != expected_tokens:
        raise RuntimeError(
            f"{engine} did not produce the forced token count "
            f"{expected_tokens}: got {len(token_ids)}"
        )
    return token_ids


def run_target_oracle(
    bundle: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 1024,
) -> tuple[int, ...]:
    output = _generate_ar(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
    )
    return _exact_tokens(
        output.tokens,
        expected_tokens=max_tokens,
        engine="target-only oracle",
    )


def arm_receipt_from_mtplx(output: Any) -> dict[str, Any]:
    stats = output.stats
    return {
        "tokens": tuple(int(token) for token in output.tokens),
        "generated_tokens": int(stats.generated_tokens),
        "decode_tps": float(stats.decode_tok_s),
        "elapsed_s": float(stats.elapsed_s),
        "decode_elapsed_s": float(stats.decode_elapsed_s),
        "peak_memory_gb": float(stats.peak_memory_bytes) / (1024**3),
        "verify_calls": int(stats.verify_calls),
        "accepted_by_depth": [int(value) for value in stats.accepted_by_depth],
        "engine": "mtplx_mtp",
    }


def run_mtp_control(
    bundle: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    output = _generate_mtpk(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        speculative_depth=MTP_DEPTH,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        mtp_cache_policy="persistent",
        mtp_history_policy="cycle",
    )
    receipt = arm_receipt_from_mtplx(output)
    receipt["tokens"] = _exact_tokens(
        receipt["tokens"],
        expected_tokens=max_tokens,
        engine="MTPLX MTP control",
    )
    if receipt["generated_tokens"] != max_tokens:
        raise RuntimeError(
            "MTPLX MTP control stats did not report the forced token count "
            f"{max_tokens}: got {receipt['generated_tokens']}"
        )
    return receipt


def build_fixed_dflash_runtime_context():
    return _build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
    )


def arm_receipt_from_dflash_events(
    events: Iterable[Any],
    *,
    requested_width: int,
    expected_tokens: int,
) -> dict[str, Any]:
    from dflash_mlx.engine.events import SummaryEvent

    summaries = [event for event in events if isinstance(event, SummaryEvent)]
    if len(summaries) != 1:
        raise RuntimeError("DFlash2 stream ended without exactly one summary")
    summary = summaries[0]

    effective_width = int(summary.block_tokens or 0)
    if effective_width != requested_width:
        raise RuntimeError(
            f"DFlash2 requested width {requested_width} became {effective_width}"
        )
    if summary.fallback_ar:
        raise RuntimeError(
            "DFlash2 reported fallback AR: "
            f"{summary.fallback_reason or 'unspecified reason'}"
        )
    generated_tokens = int(summary.generation_tokens)
    if generated_tokens != expected_tokens:
        raise RuntimeError(
            "DFlash2 did not produce the forced token count "
            f"{expected_tokens}: got {generated_tokens}"
        )
    token_ids = _exact_tokens(
        summary.generated_token_ids,
        expected_tokens=expected_tokens,
        engine="DFlash2 token ID count",
    )
    prefill_us = float(summary.phase_timings_us.get("prefill", 0.0))
    elapsed_us = float(summary.elapsed_us)
    decode_us = elapsed_us - prefill_us
    if not math.isfinite(decode_us) or decode_us <= 0.0:
        raise RuntimeError("DFlash2 summary must report a positive decode duration")

    return {
        "tokens": token_ids,
        "generated_tokens": generated_tokens,
        "decode_tps": generated_tokens / (decode_us / 1_000_000.0),
        "elapsed_s": elapsed_us / 1_000_000.0,
        "prefill_s": prefill_us / 1_000_000.0,
        "decode_elapsed_s": decode_us / 1_000_000.0,
        "peak_memory_gb": float(summary.peak_memory_gb or 0.0),
        "cycles_completed": int(summary.cycles_completed),
        "accepted_from_draft": int(summary.accepted_from_draft),
        "acceptance_ratio": float(summary.acceptance_ratio),
        "acceptance_history": [int(value) for value in summary.acceptance_history],
        "requested_width": int(requested_width),
        "effective_width": effective_width,
        "fallback_ar": bool(summary.fallback_ar),
        "fallback_reason": summary.fallback_reason,
        "engine": "dflash_mlx_0_1_10",
    }


def run_dflash2_candidate(
    bundle: Any,
    prompt_ids: Sequence[int],
    width: int,
    runtime_context: Any,
    *,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    events = _stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=list(prompt_ids),
        prompt="",
        use_chat_template=False,
        max_new_tokens=max_tokens,
        block_tokens=int(width),
        stop_token_ids=[],
        runtime_context=runtime_context,
    )
    return arm_receipt_from_dflash_events(
        events,
        requested_width=int(width),
        expected_tokens=max_tokens,
    )


def _token_sha256(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _receipt_without_tokens(arm: dict[str, Any]) -> dict[str, Any]:
    public = dict(arm)
    try:
        tokens = tuple(int(token) for token in public.pop("tokens"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("benchmark arm receipt must contain integer tokens") from error
    public["token_sha256"] = _token_sha256(tokens)
    return public


def _arm_matches_oracle(
    arm: dict[str, Any],
    oracle_tokens: tuple[int, ...],
    *,
    expected_tokens: int,
) -> bool:
    try:
        tokens = tuple(int(token) for token in arm["tokens"])
        generated_tokens = int(arm["generated_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    return tokens == oracle_tokens and generated_tokens == expected_tokens


def run_dflash2_depth_sweep(
    *,
    bundle: Any,
    prompt_ids: Sequence[int],
    widths: Sequence[int],
    repetitions: int,
    max_tokens: int = 1024,
    oracle_tokens: Sequence[int] | None = None,
    arm_runner: Any | None = None,
) -> dict[str, Any]:
    width_tuple = tuple(widths)
    if not width_tuple:
        raise ValueError("widths must not be empty")
    if any(type(width) is not int or not 1 <= width <= 8 for width in width_tuple):
        raise ValueError("widths must be integers between 1 and 8")
    if len(width_tuple) != len(set(width_tuple)):
        raise ValueError("widths must be unique")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    prompt_tuple = tuple(int(token) for token in prompt_ids)
    if not prompt_tuple:
        raise ValueError("prompt_ids must not be empty")

    if oracle_tokens is None:
        oracle_tuple = run_target_oracle(
            bundle,
            prompt_tuple,
            max_tokens=max_tokens,
        )
    else:
        oracle_tuple = _exact_tokens(
            oracle_tokens,
            expected_tokens=max_tokens,
            engine="injected target-only oracle",
        )

    production_runner = arm_runner is None
    runtime_context = None
    if production_runner:
        runtime_context = build_fixed_dflash_runtime_context()

        def resolved_arm_runner(kind: str, width: int) -> dict[str, Any]:
            if kind == "mtp":
                return run_mtp_control(
                    bundle,
                    prompt_tuple,
                    max_tokens=max_tokens,
                )
            return run_dflash2_candidate(
                bundle,
                prompt_tuple,
                width,
                runtime_context,
                max_tokens=max_tokens,
            )

    else:
        resolved_arm_runner = arm_runner

    brackets: list[dict[str, Any]] = []
    selection_rows: list[DepthBracket] = []
    warmed_widths: set[int] = set()
    for repetition in range(repetitions):
        offset = repetition % len(width_tuple)
        rotated_widths = width_tuple[offset:] + width_tuple[:offset]
        for width in rotated_widths:
            if production_runner and width not in warmed_widths:
                run_dflash2_candidate(
                    bundle,
                    prompt_tuple,
                    width,
                    runtime_context,
                    max_tokens=32,
                )
                warmed_widths.add(width)

            control_before = resolved_arm_runner("mtp", MTP_DEPTH)
            candidate = resolved_arm_runner("dflash2", width)
            control_after = resolved_arm_runner("mtp", MTP_DEPTH)
            parity_passed = all(
                _arm_matches_oracle(
                    arm,
                    oracle_tuple,
                    expected_tokens=max_tokens,
                )
                for arm in (control_before, candidate, control_after)
            )
            parity_passed = parity_passed and (
                candidate.get("requested_width") == width
                and candidate.get("effective_width") == width
                and candidate.get("fallback_ar") is False
            )

            selection_rows.append(
                DepthBracket(
                    width=width,
                    candidate_decode_tps=float(candidate["decode_tps"]),
                    control_before_tps=float(control_before["decode_tps"]),
                    control_after_tps=float(control_after["decode_tps"]),
                    parity_passed=parity_passed,
                )
            )
            brackets.append(
                {
                    "repetition": repetition,
                    "width": width,
                    "control_before": _receipt_without_tokens(control_before),
                    "candidate": _receipt_without_tokens(candidate),
                    "control_after": _receipt_without_tokens(control_after),
                    "parity_passed": parity_passed,
                }
            )

    selection = None
    if all(row.parity_passed for row in selection_rows):
        selection = asdict(select_stock_depth(selection_rows))
    return {
        "workload": {
            "prompt_tokens": len(prompt_tuple),
            "generated_tokens": max_tokens,
            "greedy": True,
        },
        "widths": list(width_tuple),
        "repetitions": repetitions,
        "oracle_token_sha256": _token_sha256(oracle_tuple),
        "brackets": brackets,
        "selection": selection,
    }
