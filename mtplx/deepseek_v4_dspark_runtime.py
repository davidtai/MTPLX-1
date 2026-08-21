"""Construction-time binding for the fixed DeepSeek V4 DSpark lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from mtplx.deepseek_v4_dspark_artifact import DSparkConfig, VerifiedDSparkArtifact
from mtplx.deepseek_v4_dspark_generation import DSparkTargetVerification
from mtplx.models.deepseek_v4 import DeepseekV4AffineInt4Cache
from mtplx.models.deepseek_v4_dspark import DeepseekV4DSparkCache


@dataclass(frozen=True)
class DeepseekV4DSparkRuntime:
    config: DSparkConfig
    make_target_cache: Callable[[], list]
    make_dspark_cache: Callable[[], list]
    target_prefill: Callable[[mx.array, list], DSparkTargetVerification]
    prefill_dspark: Callable[[tuple[mx.array, mx.array, mx.array], list], None]
    target_m6: Callable[[mx.array, list], DSparkTargetVerification]
    proposal_k5: Callable[[mx.array, list, int], mx.array]
    trim_target: Callable[[list, int], None]
    commit_dspark: Callable[[tuple[mx.array, mx.array, mx.array], list, int], None]


def install_deepseek_v4_dspark_runtime(
    model,
    artifact: VerifiedDSparkArtifact,
) -> DeepseekV4DSparkRuntime:
    """Validate fixed ownership once, then bind direct K5/M6 callables."""

    owner = model.dspark
    if owner is None or len(owner.stages) != 3:
        raise ValueError("the loaded model does not own exactly three DSpark stages")

    target_probe = model.make_cache()
    draft_probe = model.make_dspark_cache()
    if not target_probe or not all(
        isinstance(cache, DeepseekV4AffineInt4Cache) for cache in target_probe
    ):
        raise ValueError("DSpark requires affine-int4 target K/V from offset zero")
    if len(draft_probe) != 3 or not all(
        isinstance(cache, DeepseekV4DSparkCache) for cache in draft_probe
    ):
        raise ValueError("DSpark requires three affine-int4 stage K/V caches")

    def target_m6(input_ids: mx.array, target_cache: list):
        logits, taps = model(
            input_ids,
            cache=target_cache,
            return_hidden=True,
        )
        return DSparkTargetVerification(logits=logits, taps=taps)

    def target_prefill(input_ids: mx.array, target_cache: list):
        logits, taps = model(
            input_ids,
            cache=target_cache,
            return_hidden=True,
            logits_keep=1,
        )
        return DSparkTargetVerification(logits=logits, taps=taps)

    def prefill_dspark(target_taps, draft_caches: list) -> None:
        model.prefill_dspark(target_taps, draft_caches)

    def proposal_k5(
        primary_token_ids: mx.array,
        draft_caches: list,
        start_pos: int,
    ) -> mx.array:
        return model.propose_dspark_k5(
            primary_token_ids,
            draft_caches,
            start_pos=start_pos,
        ).future_tokens

    def trim_target(target_caches: list, count: int) -> None:
        for cache in target_caches:
            cache.trim(count)

    def commit_dspark(target_taps, draft_caches: list, start_pos: int) -> None:
        model.commit_dspark_main(
            target_taps,
            draft_caches,
            start_pos=start_pos,
        )

    return DeepseekV4DSparkRuntime(
        config=artifact.config,
        make_target_cache=model.make_cache,
        make_dspark_cache=model.make_dspark_cache,
        target_prefill=target_prefill,
        prefill_dspark=prefill_dspark,
        target_m6=target_m6,
        proposal_k5=proposal_k5,
        trim_target=trim_target,
        commit_dspark=commit_dspark,
    )
