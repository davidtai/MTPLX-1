"""Fixed greedy K5 / physical-M6 cycle for DeepSeek V4 DSpark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten


@dataclass(frozen=True)
class DSparkTargetVerification:
    """Target outputs produced by one physical six-row verification call."""

    logits: mx.array
    taps: tuple[mx.array, mx.array, mx.array]


@dataclass(frozen=True)
class DSparkCycleResult:
    primary: mx.array
    future_tokens: mx.array
    verify_ids: mx.array
    committed_tokens: mx.array
    next_primary: mx.array
    next_target_logits: mx.array
    accepted_future_tokens: int
    physical_verify_width: int


class DeepseekV4DSparkCycle:
    """Prebound branch-free DSpark lane, apart from causal acceptance length."""

    def __init__(
        self,
        *,
        propose_k5: Callable[[mx.array, int], mx.array],
        verify_m6: Callable[[mx.array], DSparkTargetVerification],
        trim_target: Callable[[int], None],
        commit_dspark: Callable[
            [tuple[mx.array, mx.array, mx.array], int],
            None,
        ],
    ) -> None:
        self._propose_k5 = propose_k5
        self._verify_m6 = verify_m6
        self._trim_target = trim_target
        self._commit_dspark = commit_dspark

    def __call__(self, carried_target_logits: mx.array, *, start_pos: int):
        primary = mx.argmax(carried_target_logits, axis=-1).astype(mx.int32)
        future_tokens = self._propose_k5(primary, int(start_pos))
        verify_ids = mx.concatenate([primary[:, None], future_tokens], axis=1)
        verified = self._verify_m6(verify_ids)

        target_predictions = mx.argmax(verified.logits[:, :-1], axis=-1)
        matches = (target_predictions == future_tokens).astype(mx.int32)
        accepted = int(mx.sum(mx.cumprod(matches, axis=1)).item())
        retained = accepted + 1
        rejected = int(future_tokens.shape[1]) - accepted
        if rejected:
            self._trim_target(rejected)
        retained_taps = tuple(tap[:, :retained] for tap in verified.taps)
        self._commit_dspark(retained_taps, int(start_pos))
        next_primary = mx.argmax(verified.logits[:, accepted], axis=-1).astype(
            primary.dtype
        )

        return DSparkCycleResult(
            primary=primary,
            future_tokens=future_tokens,
            verify_ids=verify_ids,
            committed_tokens=verify_ids[:, :retained],
            next_primary=next_primary,
            next_target_logits=verified.logits[:, accepted],
            accepted_future_tokens=accepted,
            physical_verify_width=int(verify_ids.shape[1]),
        )


def _cache_arrays(caches: list) -> list[mx.array]:
    arrays = []
    for cache in caches:
        for _name, value in tree_flatten(cache.state):
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def generate_dspark(
    rt,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
):
    """Run the isolated greedy K5 lane with one physical M6 per cycle."""

    import time

    from mtplx.generation import (
        GenerationOutput,
        GenerationStats,
        _decode,
        _default_stop_tokens,
        _generation_rate_fields,
        _strip_terminal_stop,
    )

    installed = rt.deepseek_v4_dspark_runtime
    if installed is None:
        raise RuntimeError("the runtime was not constructed with DSpark enabled")
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer)
        if stop_token_ids is None
        else stop_token_ids
    )
    target_cache = installed.make_target_cache()
    draft_caches = installed.make_dspark_cache()

    started = time.perf_counter()
    prompt_started = time.perf_counter()
    prompt = mx.array([prompt_ids], dtype=mx.int32)
    prefetched = installed.target_prefill(prompt, target_cache)
    installed.prefill_dspark(prefetched.taps, draft_caches)
    mx.eval(
        prefetched.logits,
        *_cache_arrays(target_cache),
        *_cache_arrays(draft_caches),
    )
    prompt_time = time.perf_counter() - prompt_started

    cycle = DeepseekV4DSparkCycle(
        propose_k5=lambda primary, start_pos: installed.proposal_k5(
            primary,
            draft_caches,
            start_pos,
        ),
        verify_m6=lambda input_ids: installed.target_m6(input_ids, target_cache),
        trim_target=lambda count: installed.trim_target(target_cache, count),
        commit_dspark=lambda taps, start_pos: installed.commit_dspark(
            taps,
            draft_caches,
            start_pos,
        ),
    )
    carried_logits = prefetched.logits[:, -1]
    start_pos = len(prompt_ids)
    tokens: list[int] = []
    accepted_by_depth = [0] * 5
    accepted_total = 0
    cycles = 0
    terminated = False
    while len(tokens) < int(max_tokens) and not terminated:
        result = cycle(carried_logits, start_pos=start_pos)
        mx.eval(
            result.committed_tokens,
            result.next_target_logits,
            *_cache_arrays(target_cache),
            *_cache_arrays(draft_caches),
        )
        accepted = int(result.accepted_future_tokens)
        accepted_total += accepted
        cycles += 1
        for depth in range(accepted):
            accepted_by_depth[depth] += 1

        committed = [int(value) for value in result.committed_tokens[0].tolist()]
        remaining = int(max_tokens) - len(tokens)
        committed = committed[:remaining]
        stop_at = next(
            (
                index
                for index, token in enumerate(committed)
                if token in stop_token_ids
            ),
            None,
        )
        if stop_at is not None:
            committed = committed[: stop_at + 1]
            terminated = True
        tokens.extend(committed)
        if token_callback is not None:
            emitted = [token for token in committed if token not in stop_token_ids]
            if emitted:
                token_callback(emitted)
        start_pos += int(result.committed_tokens.shape[1])
        carried_logits = result.next_target_logits

    elapsed = time.perf_counter() - started
    rates = _generation_rate_fields(
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        prompt_eval_time_s=prompt_time,
    )
    accepted_depth_histogram = {
        str(depth): (
            cycles - accepted_by_depth[0]
            if depth == 0
            else accepted_by_depth[depth - 1]
            - (accepted_by_depth[depth] if depth < 5 else 0)
        )
        for depth in range(6)
    }
    stats = GenerationStats(
        mode="dspark",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        prompt_eval_time_s=prompt_time,
        prompt_tps=(len(prompt_ids) / prompt_time if prompt_time > 0 else 0.0),
        accepted_drafts=accepted_total,
        rejected_drafts=cycles * 5 - accepted_total,
        drafted_tokens=cycles * 5,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=[cycles] * 5,
        verify_calls=cycles,
        verify_hidden_mode="dspark_taps_40_41_42",
        events=[
            {
                "physical_verify_width": 6,
                "verify_width_histogram": {"6": cycles},
                "accepted_depth_histogram": accepted_depth_histogram,
            }
        ],
        **rates,
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
        stats=stats,
        finish_reason="stop" if terminated else "length",
    )
