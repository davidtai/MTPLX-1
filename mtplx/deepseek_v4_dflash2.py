"""DeepSeek V4 DSpark adapters for the existing DFlash2 engine."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.model import DraftRuntimeCapabilities

from mtplx.models.deepseek_v4 import DeepseekV4AffineInt4Cache


_TARGET_LAYER_IDS = (40, 41, 42)
_CAPTURE_LAYER_IDS = tuple(layer_id + 1 for layer_id in _TARGET_LAYER_IDS)
_PHYSICAL_VERIFY_WIDTH = 6


class DeepseekV4TargetOps:
    """Bind a construction-qualified DeepSeek V4 target to DFlash2."""

    backend_name = "deepseek_v4_dspark"

    def model_type(self, target_model: Any) -> str:
        return str(getattr(getattr(target_model, "args", None), "model_type", "")).lower()

    def supports_model(self, target_model: Any) -> bool:
        stages = tuple(getattr(getattr(target_model, "dspark", None), "stages", ()))
        layer_ids = tuple(
            int(value)
            for value in (
                getattr(
                    getattr(target_model, "args", None),
                    "dspark_target_layer_ids",
                    (),
                )
                or ()
            )
        )
        return (
            self.model_type(target_model) == "deepseek_v4"
            and len(stages) == 3
            and layer_ids == _TARGET_LAYER_IDS
        )

    def family(self, target_model: Any) -> str:
        del target_model
        return self.backend_name

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=False,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            supports_verify_linear=True,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_model(self, target_model: Any) -> Any:
        return target_model.model

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self,
        target_model: Any,
        hidden_states: mx.array,
    ) -> mx.array:
        return target_model.lm_head(hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        del enable_speculative_linear_cache
        if quantize_kv_cache:
            raise ValueError(
                "DeepSeek V4 target K/V is already affine-int4 from offset zero"
            )
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("DeepSeek V4 uses its model-defined attention windows")
        cache = target_model.make_cache()
        if not cache or not all(
            isinstance(entry, DeepseekV4AffineInt4Cache) for entry in cache
        ):
            raise ValueError("DeepSeek V4 DFlash2 requires affine-int4 target caches")
        return cache

    def install_speculative_hooks(self, target_model: Any) -> None:
        del target_model

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        if input_embeddings is not None:
            raise ValueError("DeepSeek V4 DFlash2 does not support input embeddings")
        if input_ids is None:
            raise ValueError("DeepSeek V4 DFlash2 requires input IDs")
        logits, taps = target_model(
            input_ids,
            cache=cache,
            return_hidden=True,
            logits_keep=1 if logits_last_only else None,
        )
        if len(taps) != len(_TARGET_LAYER_IDS):
            raise RuntimeError("DeepSeek V4 target did not return taps 40/41/42")
        captured = dict(zip(_CAPTURE_LAYER_IDS, taps, strict=True))
        if capture_layer_ids is not None:
            captured = {
                layer_id: value
                for layer_id, value in captured.items()
                if layer_id in capture_layer_ids
            }
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("DeepSeek V4 DSpark does not support DDTree")

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("DeepSeek V4 DSpark does not support DDTree")

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        layer_ids = tuple(int(value) for value in target_layer_ids)
        if layer_ids != _TARGET_LAYER_IDS:
            raise ValueError(
                "DeepSeek V4 DSpark target layer IDs must be exactly 40/41/42"
            )
        return mx.concatenate(
            [captured_dict[layer_id + 1] for layer_id in layer_ids],
            axis=-1,
        )

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        del cache_entries, prefix_len

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        del acceptance_length, drafted_tokens
        started = time.perf_counter_ns()
        for cache_entry in cache_entries:
            offset = int(cache_entry.offset)
            if offset < int(target_len):
                raise ValueError(
                    "DeepSeek V4 target cache ended before the accepted prefix"
                )
            if offset > int(target_len):
                cache_entry.trim(offset - int(target_len))
        return time.perf_counter_ns() - started

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        draft_cache.clear()
        target_cache.clear()


class DeepseekV4DSparkDraftAdapter:
    """Expose the installed K5 DSpark owner through DFlash2's draft metadata."""

    def __init__(self, target_model: Any) -> None:
        owner = getattr(target_model, "dspark", None)
        stages = tuple(getattr(owner, "stages", ()))
        if len(stages) != 3:
            raise ValueError("DeepSeek V4 DFlash2 requires exactly three DSpark stages")
        target_layer_ids = tuple(
            int(value)
            for value in (
                getattr(target_model.args, "dspark_target_layer_ids", ()) or ()
            )
        )
        if target_layer_ids != _TARGET_LAYER_IDS:
            raise ValueError("DeepSeek V4 DFlash2 requires target taps 40/41/42")

        self.target_model = target_model
        self.owner = owner
        self.target_layer_ids = list(_TARGET_LAYER_IDS)
        self.block_size = _PHYSICAL_VERIFY_WIDTH
        self.mask_token_id = int(target_model.args.dspark_noise_token_id)
        self.args = SimpleNamespace(
            sliding_window=max(int(stage.attn.window_size) for stage in stages),
            layer_types=("sliding_attention",) * len(stages),
        )
        self.capabilities = DraftRuntimeCapabilities(
            default_block_tokens=_PHYSICAL_VERIFY_WIDTH,
            max_block_tokens=_PHYSICAL_VERIFY_WIDTH,
            supports_copyspec=False,
            supports_ddtree=False,
            supports_early_rollback_launch=False,
        )

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        if target_model is not self.target_model or not isinstance(
            target_ops, DeepseekV4TargetOps
        ):
            raise ValueError("DSpark draft adapter is bound to one DeepSeek target")

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        hidden_size = int(self.target_model.args.hidden_size)
        expected_width = hidden_size * len(_TARGET_LAYER_IDS)
        if int(target_hidden.shape[-1]) != expected_width:
            raise ValueError(
                "DeepSeek V4 captured target width does not match taps 40/41/42"
            )
        taps = tuple(
            target_hidden[..., index * hidden_size : (index + 1) * hidden_size]
            for index in range(len(_TARGET_LAYER_IDS))
        )
        return self.owner.stages[0].fuse_main(taps)


class DeepseekV4DSparkBackend:
    """Append accepted target context and invoke the installed DSpark K5 model."""

    def make_cache(
        self,
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        sink_size: int,
        window_size: int,
        allow_full_context_layers: bool = False,
    ) -> list[Any]:
        del sink_size, window_size
        if allow_full_context_layers:
            raise ValueError("DSpark stages use their fixed sliding attention window")
        caches = draft_model.owner.make_cache()
        if len(caches) != 3:
            raise ValueError("DeepSeek V4 DFlash2 requires three DSpark caches")
        for cache in caches:
            ring = getattr(cache, "ring", None)
            if (
                getattr(ring, "bits", None) != 4
                or getattr(ring, "group_size", None) != 64
                or len(ring) != 0
            ):
                raise ValueError(
                    "DSpark DFlash2 caches must start empty in affine-int4 group64"
                )
        return caches

    @staticmethod
    def _append_context(
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        draft_context: mx.array,
    ) -> int:
        if len(draft_cache) != 3 or int(draft_context.ndim) != 3:
            raise ValueError("DSpark context append requires three caches and rank-3 rows")
        lengths = tuple(int(cache.prefill_length) for cache in draft_cache)
        if len(set(lengths)) != 1:
            raise RuntimeError("DSpark stage cache positions diverged")
        prior_length = lengths[0]
        context_rows = int(draft_context.shape[1])
        if context_rows <= 0:
            raise ValueError("DSpark context append requires at least one target row")

        if prior_length == 0:
            for stage, cache in zip(
                draft_model.owner.stages,
                draft_cache,
                strict=True,
            ):
                stage.attn.prefill_context(draft_context, cache)
        else:
            positions = mx.arange(prior_length, prior_length + context_rows)
            for stage, cache in zip(
                draft_model.owner.stages,
                draft_cache,
                strict=True,
            ):
                cache.commit_main(
                    prior_length,
                    stage.attn.project_kv(draft_context, positions),
                )
        return prior_length + context_rows

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        del target_ops, mask_token_tail
        if target_model is not draft_model.target_model:
            raise ValueError("DSpark backend received a different target model")
        requested_width = int(block_len)
        if not 2 <= requested_width <= _PHYSICAL_VERIFY_WIDTH:
            raise ValueError("DeepSeek V4 DSpark draft width must be between two and six")
        if suppress_token_mask is not None:
            raise ValueError("DeepSeek V4 DSpark does not support token suppression")

        start_pos = self._append_context(
            draft_model=draft_model,
            draft_cache=draft_cache,
            draft_context=draft_context,
        )
        proposal = draft_model.owner.propose_k5(
            staged_first,
            target_model.model.embed_tokens,
            target_model.lm_head,
            draft_cache,
            start_pos=start_pos,
        )
        full_draft = proposal.future_tokens.squeeze(0).astype(mx.uint32)
        if tuple(full_draft.shape) != (_PHYSICAL_VERIFY_WIDTH - 1,):
            raise ValueError("DeepSeek V4 DSpark must return exactly five future tokens")
        drafted = full_draft[: requested_width - 1]
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(drafted)
        return drafted

    def advance_context(
        self,
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        draft_context: mx.array,
    ) -> None:
        self._append_context(
            draft_model=draft_model,
            draft_cache=draft_cache,
            draft_context=draft_context,
        )


def _stream_dflash_generate(**kwargs):
    from dflash_mlx.runtime import stream_dflash_generate

    return stream_dflash_generate(**kwargs)


def generate_deepseek_v4_dflash2(
    bundle: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    runtime_context: Any,
    stop_token_ids: Optional[list[int]] = None,
    token_callback: Any = None,
):
    """Translate the unchanged DFlash2 event stream into MTPLX output types."""

    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids
    from mtplx.generation import GenerationOutput, GenerationStats

    if runtime_context is None:
        raise ValueError("DeepSeek V4 DFlash2 requires a prebuilt runtime context")
    resolved_stop_ids = (
        [int(value) for value in stop_token_ids]
        if stop_token_ids is not None
        else get_stop_token_ids(bundle.tokenizer)
    )
    stop_set = set(resolved_stop_ids)
    summary = None
    stop_seen = False
    for event in _stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=[int(value) for value in prompt_ids],
        prompt="",
        use_chat_template=False,
        max_new_tokens=int(max_tokens),
        block_tokens=_PHYSICAL_VERIFY_WIDTH,
        stop_token_ids=resolved_stop_ids,
        quantize_kv_cache=False,
        runtime_context=runtime_context,
    ):
        if isinstance(event, TokenEvent):
            token_id = int(event.token_id)
            if token_id in stop_set:
                stop_seen = True
            elif token_callback is not None and not stop_seen:
                token_callback([token_id])
        elif isinstance(event, SummaryEvent):
            if summary is not None:
                raise RuntimeError("DFlash2 emitted more than one summary")
            summary = event

    if summary is None:
        raise RuntimeError("DFlash2 stream ended without a summary")
    if summary.fallback_ar:
        raise RuntimeError(
            "DeepSeek V4 DFlash2 refused its installed lane: "
            f"{summary.fallback_reason or 'unspecified fallback'}"
        )
    if int(summary.block_tokens or 0) != _PHYSICAL_VERIFY_WIDTH:
        raise RuntimeError("DeepSeek V4 DFlash2 did not execute physical M6")

    physical_tokens = [int(value) for value in summary.generated_token_ids]
    first_stop = next(
        (index for index, token_id in enumerate(physical_tokens) if token_id in stop_set),
        None,
    )
    tokens = physical_tokens if first_stop is None else physical_tokens[:first_stop]
    elapsed_s = float(summary.elapsed_us) / 1_000_000.0
    prompt_s = float(summary.phase_timings_us.get("prefill", 0.0)) / 1_000_000.0
    decode_s = max(0.0, elapsed_s - prompt_s)
    cycles = int(summary.cycles_completed)
    acceptance_history = tuple(int(value) for value in summary.acceptance_history)
    accepted_by_depth = [
        sum(1 for accepted in acceptance_history if accepted >= depth)
        for depth in range(1, _PHYSICAL_VERIFY_WIDTH)
    ]
    accepted = int(summary.accepted_from_draft)
    generated_before_cycle = 0
    drafted_per_cycle = []
    for acceptance_len in acceptance_history:
        remaining = max(0, int(max_tokens) - generated_before_cycle)
        verify_width = min(_PHYSICAL_VERIFY_WIDTH, remaining)
        drafted_per_cycle.append(max(0, verify_width - 1))
        generated_before_cycle += min(remaining, 1 + acceptance_len)
    drafted = sum(drafted_per_cycle)
    drafted_by_depth = [
        sum(1 for draft_count in drafted_per_cycle if draft_count >= depth)
        for depth in range(1, _PHYSICAL_VERIFY_WIDTH)
    ]
    generated = len(tokens)
    stats = GenerationStats(
        mode="dspark",
        generated_tokens=generated,
        elapsed_s=elapsed_s,
        tok_s=(generated / elapsed_s if elapsed_s > 0 else 0.0),
        decode_elapsed_s=decode_s,
        decode_tok_s=(generated / decode_s if decode_s > 0 else 0.0),
        end_to_end_tok_s=(generated / elapsed_s if elapsed_s > 0 else 0.0),
        accepted_drafts=accepted,
        rejected_drafts=max(0, drafted - accepted),
        drafted_tokens=drafted,
        verify_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("verify", 0.0)
        )
        / 1_000_000.0,
        draft_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("draft", 0.0)
        )
        / 1_000_000.0,
        prompt_eval_time_s=prompt_s,
        prompt_tps=(
            int(summary.prompt_token_count) / prompt_s if prompt_s > 0 else 0.0
        ),
        rollback_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("rollback", 0.0)
        )
        / 1_000_000.0,
        peak_memory_bytes=int(float(summary.peak_memory_gb or 0.0) * 1_000_000_000),
        speculative_depth=_PHYSICAL_VERIFY_WIDTH - 1,
        requested_speculative_depth=_PHYSICAL_VERIFY_WIDTH - 1,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        verify_calls=cycles,
        verify_hidden_mode="dflash2_deepseek_taps_40_41_42",
        events=[summary.to_payload()],
    )
    return GenerationOutput(
        tokens=tokens,
        text=bundle.tokenizer.decode(tokens),
        stats=stats,
        final_state=None,
        finish_reason="stop" if first_stop is not None else "length",
    )
