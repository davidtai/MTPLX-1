"""Native fixed-K5 DSpark model components for DeepSeek V4 Flash."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn

from mtplx.deepseek_v4_nvfp4_kv import MiaNVFP4Rows
from mtplx.kernels.deepseek_v4_nvfp4_mla import install_nvfp4_sparse_mla
from mtplx.models.deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    HeadHC,
    ModelArgs,
    _apply_interleaved_rope,
)


DSPARK_STAGE_COUNT = 3
DSPARK_BLOCK_SIZE = 5
DSPARK_NOISE_TOKEN_ID = 128799
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
DSPARK_MARKOV_RANK = 256


class _MarkovHead(Protocol):
    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]: ...


def greedy_future_tokens(
    neural_logits: mx.array,
    primary_token_ids: mx.array,
    markov_head: _MarkovHead,
) -> mx.array:
    """Produce five genuinely future tokens, conditioning row zero on primary."""

    if tuple(neural_logits.shape[:2]) != (
        int(primary_token_ids.shape[0]),
        DSPARK_BLOCK_SIZE,
    ):
        raise ValueError("DSpark K5 logits must have shape [batch, 5, vocab]")
    previous = primary_token_ids
    future = []
    for row in range(DSPARK_BLOCK_SIZE):
        bias, _markov_embedding = markov_head(previous)
        previous = mx.argmax(neural_logits[:, row] + bias, axis=-1).astype(
            primary_token_ids.dtype
        )
        future.append(previous)
    return mx.stack(future, axis=1)


class DeepseekV4DSparkCache:
    """One stage's fixed sliding-window context ring in Mia stock432 storage."""

    def __init__(self, *, window_size: int, head_dim: int) -> None:
        self.window_size = int(window_size)
        self.head_dim = int(head_dim)
        if self.head_dim != 512:
            raise ValueError("Mia DSpark cache requires head_dim=512")
        self.ring = MiaNVFP4Rows()
        self.prefill_length = 0

    def prefill(self, main_latent: mx.array, main_rope: mx.array) -> None:
        batch, sequence_length, width = (int(v) for v in main_latent.shape)
        if width != self.head_dim:
            raise ValueError("DSpark cache head dimension mismatch")
        if tuple(main_rope.shape) != (batch, sequence_length, 64):
            raise ValueError("DSpark cache RoPE rows must have shape [batch, rows, 64]")
        if sequence_length <= self.window_size:
            latent_padding = mx.zeros(
                (batch, self.window_size - sequence_length, width),
                dtype=main_latent.dtype,
            )
            rope_padding = mx.zeros(
                (batch, self.window_size - sequence_length, 64),
                dtype=main_rope.dtype,
            )
            latent_rows = mx.concatenate([main_latent, latent_padding], axis=1)
            rope_rows = mx.concatenate([main_rope, rope_padding], axis=1)
        else:
            latent_last = main_latent[:, -self.window_size :]
            rope_last = main_rope[:, -self.window_size :]
            cutoff = sequence_length % self.window_size
            latent_rows = (
                latent_last
                if cutoff == 0
                else mx.concatenate(
                    [
                        latent_last[:, self.window_size - cutoff :],
                        latent_last[:, : self.window_size - cutoff],
                    ],
                    axis=1,
                )
            )
            rope_rows = (
                rope_last
                if cutoff == 0
                else mx.concatenate(
                    [
                        rope_last[:, self.window_size - cutoff :],
                        rope_last[:, : self.window_size - cutoff],
                    ],
                    axis=1,
                )
            )
        self.ring.clear()
        self.ring.append(latent_rows, rope_rows)
        self.prefill_length = sequence_length

    def visible_rows(self) -> tuple[mx.array, mx.array]:
        if len(self.ring) != self.window_size:
            raise RuntimeError("DSpark attention cache has not been prefetched")
        return self.ring.decode()

    def commit_main(
        self,
        start_pos: int,
        main_latent: mx.array,
        main_rope: mx.array,
    ) -> None:
        if len(self.ring) != self.window_size:
            raise RuntimeError("DSpark decode requires attention-only prefill first")
        if (
            main_latent.ndim != 3
            or main_rope.ndim != 3
            or int(main_latent.shape[0]) != int(self.ring.shape[0])
            or tuple(main_latent.shape[:-1]) != tuple(main_rope.shape[:-1])
            or int(main_rope.shape[-1]) != 64
        ):
            raise ValueError("DSpark committed main K/V must match the ring batch")
        count = int(main_latent.shape[1])
        if count <= 0 or count > self.window_size:
            raise ValueError("DSpark committed main K/V width is outside its ring")
        start = int(start_pos) % self.window_size
        first = min(count, self.window_size - start)
        self.ring.replace(
            start,
            main_latent[:, :first],
            main_rope[:, :first],
        )
        if first < count:
            self.ring.replace(0, main_latent[:, first:], main_rope[:, first:])
        self.prefill_length = max(self.prefill_length, int(start_pos) + count)


class DSparkTargetRoute:
    """Construction-bound target route exposing ordered post-layer HC means."""

    def __init__(self, target_layer_ids=DSPARK_TARGET_LAYER_IDS) -> None:
        self.target_layer_ids = tuple(int(layer_id) for layer_id in target_layer_ids)

    def __call__(self, owner, inputs: mx.array, cache):
        hidden = owner.model.embed_tokens(inputs)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], owner.args.hc_mult, hidden.shape[-1]),
        )
        if cache is None:
            cache = [None] * len(owner.model.layers)
        taps = []
        for layer_id, (layer, layer_cache) in enumerate(zip(owner.model.layers, cache)):
            hidden = layer(
                hidden,
                mask=None,
                cache=layer_cache,
                input_ids=inputs,
            )
            if layer_id in self.target_layer_ids:
                taps.append(mx.mean(hidden, axis=2))
        if len(taps) != len(self.target_layer_ids):
            raise RuntimeError("DSpark target route did not observe every required tap")
        return hidden, tuple(taps)


def _validate_dspark_args(args) -> None:
    observed = (
        int(args.dspark_block_size or 0),
        int(args.dspark_noise_token_id or 0),
        tuple(int(value) for value in (args.dspark_target_layer_ids or ())),
        int(args.dspark_markov_rank or 0),
        int(args.num_nextn_predict_layers),
    )
    expected = (
        DSPARK_BLOCK_SIZE,
        DSPARK_NOISE_TOKEN_ID,
        DSPARK_TARGET_LAYER_IDS,
        DSPARK_MARKOV_RANK,
        1,
    )
    if observed != expected:
        raise ValueError(
            f"unsupported DeepSeek V4 DSpark contract: observed={observed!r}, "
            f"expected={expected!r}"
        )
    if int(args.num_hidden_layers) <= DSPARK_TARGET_LAYER_IDS[-1]:
        raise ValueError("DeepSeek V4 DSpark target taps are absent from the trunk")
    if int(args.vocab_size) <= DSPARK_NOISE_TOKEN_ID:
        raise ValueError("DeepSeek V4 DSpark vocabulary omits the noise token")
    ratios = tuple(int(value) for value in args.compress_ratios)
    for layer_id in range(
        int(args.num_hidden_layers),
        int(args.num_hidden_layers) + DSPARK_STAGE_COUNT,
    ):
        if layer_id < len(ratios) and ratios[layer_id] != 0:
            raise ValueError("DeepSeek V4 DSpark stages require uncompressed attention")


def _dspark_visibility_indices(
    window_size: int,
    block_size: int,
    start_pos: int,
) -> mx.array:
    if int(start_pos) <= 0:
        raise ValueError("DSpark decode visibility requires start_pos > 0")
    main = mx.arange(min(int(window_size), int(start_pos)), dtype=mx.int32)
    draft = int(window_size) + mx.arange(int(block_size), dtype=mx.int32)
    return mx.concatenate([main, draft])


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """Official DSpark attention with a Mia stock432 stage context ring."""

    def __init__(self, args: ModelArgs, layer_id: int) -> None:
        super().__init__(args, layer_id)
        if self.compress_ratio != 0:
            raise ValueError("DSpark attention requires compress_ratio=0")
        self._nvfp4_sparse_mla = install_nvfp4_sparse_mla(
            heads=self.n_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            window_size=self.window_size,
        )

    def project_kv(
        self,
        hidden: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        rope_dim = self.rope_head_dim
        cos, sin = self._rope_tables(positions)
        latent = self.kv_norm(self.wkv(hidden))
        rope = _apply_interleaved_rope(
            latent[..., -rope_dim:],
            cos[None],
            sin[None],
        )
        return latent, rope

    def prefill_context(
        self,
        main_hidden: mx.array,
        cache: DeepseekV4DSparkCache,
    ) -> None:
        positions = mx.arange(int(main_hidden.shape[1]))
        cache.prefill(*self.project_kv(main_hidden, positions))

    def __call__(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
    ) -> mx.array:
        if int(start_pos) <= 0:
            raise ValueError("DSpark decode attention requires a positive position")
        batch, block, _ = hidden.shape
        if int(block) != DSPARK_BLOCK_SIZE:
            raise ValueError("DSpark decode requires five neural rows")

        positions = mx.arange(int(start_pos), int(start_pos) + block)
        cos, sin = self._rope_tables(positions)
        rope_dim = self.rope_head_dim

        query_rank = self.q_norm(self.wq_a(hidden))
        query = self.wq_b(query_rank).reshape(
            batch,
            block,
            self.n_heads,
            self.head_dim,
        )
        query = query * mx.rsqrt(
            mx.mean(mx.square(query.astype(mx.float32)), axis=-1, keepdims=True)
            + self.eps
        )
        query = query.astype(hidden.dtype)
        query = mx.concatenate(
            [
                query[..., :-rope_dim],
                _apply_interleaved_rope(
                    query[..., -rope_dim:],
                    cos[None, :, None],
                    sin[None, :, None],
                ),
            ],
            axis=-1,
        )

        draft_latent, draft_rope = self.project_kv(hidden, positions)
        draft_rows = MiaNVFP4Rows()
        draft_rows.append(draft_latent, draft_rope)
        all_records = mx.concatenate(
            [cache.ring.records, draft_rows.records],
            axis=1,
        )
        visible_indices = _dspark_visibility_indices(
            self.window_size,
            block,
            int(start_pos),
        )
        visible_records = all_records[:, visible_indices]
        output = self._nvfp4_sparse_mla(
            query.transpose(0, 2, 1, 3),
            visible_records[:, :0],
            0,
            positions.astype(mx.int32),
            visible_records,
            None,
            mx.full((1, block), visible_records.shape[1], dtype=mx.int32),
            self.attn_sink.astype(mx.float32),
            self.softmax_scale,
        )
        output = output.transpose(0, 2, 1, 3)
        return self._o_lora(
            output.reshape(batch, block, self.n_heads * self.head_dim)
        )


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embedding = self.markov_w1(token_ids)
        return self.markov_w2(embedding), embedding


class DSparkConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, markov_rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=False)


class DeepseekV4DSparkStage(DeepseekV4DecoderLayer):
    """One stage of the three-layer DSpark owner."""

    def __init__(self, args: ModelArgs, stage_id: int) -> None:
        layer_id = int(args.num_hidden_layers) + int(stage_id)
        ratios = list(args.compress_ratios)
        if len(ratios) <= layer_id:
            ratios.extend([0] * (layer_id + 1 - len(ratios)))
            args = replace(args, compress_ratios=ratios)
        super().__init__(args, layer_id)
        self.attn = DeepseekV4DSparkAttention(args, layer_id)
        self.stage_id = int(stage_id)
        self.main_proj = None
        self.main_norm = None
        self.norm = None
        self.hc_head = None
        self.markov_head = None
        self.confidence_head = None
        if self.stage_id == 0:
            self.main_proj = nn.Linear(
                args.hidden_size * len(args.dspark_target_layer_ids),
                args.hidden_size,
                bias=False,
            )
            self.main_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if self.stage_id == DSPARK_STAGE_COUNT - 1:
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hc_head = HeadHC(args.hidden_size, args.hc_mult, args.hc_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size,
                args.dspark_markov_rank,
            )
            self.confidence_head = DSparkConfidenceHead(
                args.hidden_size,
                args.dspark_markov_rank,
            )

    def fuse_main(self, target_taps: tuple[mx.array, mx.array, mx.array]) -> mx.array:
        if self.main_proj is None or self.main_norm is None:
            raise RuntimeError("DSpark target-tap fusion belongs to stage zero")
        return self.main_norm(self.main_proj(mx.concatenate(target_taps, axis=-1)))

    def __call__(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
        input_ids: mx.array,
    ) -> mx.array:
        residual = hidden
        value, post, combination = self.attn_hc.pre(hidden)
        value = self.attn_norm(value)
        value = self.attn(
            value,
            start_pos=start_pos,
            cache=cache,
        )
        hidden = self.attn_hc.post(value, residual, post, combination)

        residual = hidden
        value, post, combination = self.ffn_hc.pre(hidden)
        value = self.ffn_norm(value)
        value = self.ffn(value, input_ids=input_ids)
        return self.ffn_hc.post(value, residual, post, combination)


@dataclass(frozen=True)
class DSparkModelProposal:
    future_tokens: mx.array
    neural_logits: mx.array


class DeepseekV4DSparkOwner:
    """Three stage-owned DSpark blocks with an unambiguous future-token API."""

    def __init__(self, args, stages) -> None:
        self.args = args
        self.block_size = DSPARK_BLOCK_SIZE
        self.noise_token_id = DSPARK_NOISE_TOKEN_ID
        self.target_layer_ids = DSPARK_TARGET_LAYER_IDS
        self.stages = list(stages)
        if len(self.stages) != DSPARK_STAGE_COUNT:
            raise ValueError("DeepSeek V4 DSpark requires exactly three stages")

    def draft_input_ids(self, primary_token_ids: mx.array) -> mx.array:
        if primary_token_ids.ndim != 1:
            raise ValueError("DSpark primary ids must have shape [batch]")
        noise = mx.full(
            (primary_token_ids.shape[0], self.block_size - 1),
            self.noise_token_id,
            dtype=primary_token_ids.dtype,
        )
        return mx.concatenate([primary_token_ids[:, None], noise], axis=1)

    def make_cache(self) -> list[DeepseekV4DSparkCache]:
        return [
            DeepseekV4DSparkCache(
                window_size=stage.attn.window_size,
                head_dim=stage.attn.head_dim,
            )
            for stage in self.stages
        ]

    def prefill(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
    ) -> None:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark prefill requires one cache per stage")
        main_hidden = self.stages[0].fuse_main(target_taps)
        for stage, cache in zip(self.stages, caches):
            stage.attn.prefill_context(main_hidden, cache)

    def commit_main(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> None:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark commit requires one cache per stage")
        main_hidden = self.stages[0].fuse_main(target_taps)
        positions = mx.arange(int(start_pos), int(start_pos) + int(main_hidden.shape[1]))
        for stage, cache in zip(self.stages, caches):
            latent, rope = stage.attn.project_kv(main_hidden, positions)
            cache.commit_main(start_pos, latent, rope)

    def propose_k5(
        self,
        primary_token_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> DSparkModelProposal:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark proposal requires one cache per stage")
        input_ids = self.draft_input_ids(primary_token_ids)
        hidden = embed_tokens(input_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], self.args.hc_mult, hidden.shape[-1]),
        )
        for stage, cache in zip(self.stages, caches):
            hidden = stage(
                hidden,
                start_pos=start_pos,
                cache=cache,
                input_ids=input_ids,
            )
        final = self.stages[-1]
        if final.hc_head is None or final.norm is None or final.markov_head is None:
            raise RuntimeError("DSpark final stage does not own its output heads")
        neural_logits = lm_head(final.norm(final.hc_head(hidden)))
        return DSparkModelProposal(
            future_tokens=greedy_future_tokens(
                neural_logits,
                primary_token_ids,
                final.markov_head,
            ),
            neural_logits=neural_logits,
        )


def build_deepseek_v4_dspark(args) -> DeepseekV4DSparkOwner:
    _validate_dspark_args(args)
    return DeepseekV4DSparkOwner(
        args,
        [DeepseekV4DSparkStage(args, stage_id) for stage_id in range(DSPARK_STAGE_COUNT)],
    )
