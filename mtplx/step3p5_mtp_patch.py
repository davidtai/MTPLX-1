"""Runtime MTP injection for StepFun Step-3.7-Flash (``step3p5`` / ``step3p7``).

Step ships a DeepSeek-style *appended-layer* MTP: ``num_nextn_predict_layers``
distinct decoder layers (indices ``num_hidden_layers + i``), each composed of
``enorm`` / ``hnorm`` / ``eh_proj`` + a full Step3p5 decoder block
(``mtp_block``) + a shared head (``transformer.shared_head.{norm,output}`` in the
checkpoint).

This module exists instead of routing Step through the generic DeepSeek injector
because Step has two correctness requirements the generic path gets wrong, and
either one alone collapses acceptance to the ~3% "donkey" band documented in
vLLM issue #38339:

1. **Pre-norm hidden.** The MTP consumes the trunk's hidden state *before* the
   final norm. mlx-lm's ``Step3p5Model.__call__`` returns the *post*-norm hidden
   (it applies ``self.norm`` before returning), so the injected wrapper
   re-runs the trunk layer loop and exposes the pre-norm residual stream.
2. **Zero-centered (Gemma) norms.** Every Step norm is zero-centered. We reuse
   mlx-lm's ``ZeroCenteredRMSNorm`` and ``Step3p5DecoderLayer`` so the modules
   match the math natively, and we add ``+1.0`` to the (centered) norm weights of
   a vanilla BF16 overlay at load, exactly like mlx-lm's ``Step3p5`` sanitize
   does for vanilla checkpoints.

The MTP block is a *dense* Step3p5 decoder layer (no MoE / no expert stacking)
and Step attention is plain GQA (no MLA ``kv_b_proj`` rewrite), so the weight
remap here is materially simpler than the DeepSeek path.

The injected model exposes the same ``__call__`` / ``mtp_forward`` /
``mtp_update_cache`` / ``make_mtp_cache`` contract that ``mtplx.generation``
already drives for every native-MTP backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .artifacts import (
    candidate_mtp_weight_files,
    model_type as _model_type,
    runtime_mtp_layer_count as _num_mtp_layers,
    text_config,
)

logger = logging.getLogger(__name__)

STEP_MTP_MODEL_TYPES = {"step3p5", "step3p7"}

# Norm-weight leaves that are zero-centered in a vanilla Step checkpoint and
# therefore need the ``+1.0`` shift before a plain ZeroCenteredRMSNorm uses them.
_NORM_LEAF = "norm.weight"


def is_step3p5_mtp_config(config: dict[str, Any]) -> bool:
    """True for Step-3.5/3.7 configs that declare appended MTP layers."""
    return _model_type(config) in STEP_MTP_MODEL_TYPES and _num_mtp_layers(config) > 0


def _hidden_variant_from_config(config: dict[str, Any], contract: Any | None) -> str:
    # The verified Step contract is pre-norm (matches the vLLM/StepFun reference).
    # An explicit runtime contract or config override can still flip it for the
    # hidden-variant sweep, but pre-norm is the default that produces acceptance.
    override = config.get("mtplx_mtp_hidden_variant")
    if override in {"pre_norm", "post_norm"}:
        return str(override)
    variant = getattr(contract, "hidden_variant", None)
    # MTPContract defaults to "post_norm" for the Qwen path; only honor an
    # explicit non-default request here, otherwise use the Step default.
    if variant == "pre_norm":
        return "pre_norm"
    return "pre_norm"


def _strip_outer_prefix(key: str) -> str:
    for prefix in ("language_model.model.", "language_model.", "model.model."):
        if key.startswith(prefix):
            # Normalize to the bare "model."-relative form used below.
            if prefix == "language_model.model.":
                return "model." + key[len(prefix):]
            if prefix == "model.model.":
                return "model." + key[len(prefix):]
            return key[len(prefix):]
    return key


def _rewrite_step_mtp_weights(
    raw: dict[str, Any],
    *,
    start_layer: int,
    num_mtp_layers: int,
) -> dict[str, Any]:
    """Map checkpoint MTP keys onto the injected ``_StepMTP`` module tree.

    Accepted input key shapes (any outer ``language_model.`` prefix is stripped):
      ``model.layers.{start+i}.enorm.weight``               -> ``layers.{i}.enorm.weight``
      ``model.layers.{start+i}.hnorm.weight``               -> ``layers.{i}.hnorm.weight``
      ``model.layers.{start+i}.eh_proj.weight``             -> ``layers.{i}.eh_proj.weight``
      ``model.layers.{start+i}.transformer.shared_head.norm.*``   -> ``layers.{i}.shared_head_norm.*``
      ``model.layers.{start+i}.transformer.shared_head.output.*`` -> ``layers.{i}.shared_head_head.*``
      ``model.layers.{start+i}.<rest>``                     -> ``layers.{i}.mtp_block.<rest>``
      ``model.embed_tokens.*`` / ``embed_tokens.*``         -> dropped (shared with trunk)
    Also tolerates an ``mtp.``-namespaced sidecar (keys already relative).
    """
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        k = _strip_outer_prefix(str(key))
        if k.startswith("mtp."):
            mapped[k[len("mtp."):]] = value
            continue
        if k.startswith("layers.") and not k.startswith(tuple(
            f"layers.{start_layer + i}." for i in range(num_mtp_layers)
        )):
            # Already relative (local) MTP keys, e.g. "layers.0.mtp_block..."
            mapped[k] = value
            continue
        if k.startswith(("embed_tokens.", "model.embed_tokens.")):
            continue  # shared embedding, reused from the trunk
        for local in range(num_mtp_layers):
            spec = start_layer + local
            prefix = f"model.layers.{spec}."
            if not k.startswith(prefix):
                continue
            suffix = k[len(prefix):]
            lp = f"layers.{local}"
            if suffix.startswith("transformer.shared_head.norm."):
                tail = suffix[len("transformer.shared_head.norm."):]
                mapped[f"{lp}.shared_head_norm.{tail}"] = value
            elif suffix.startswith("transformer.shared_head.output."):
                tail = suffix[len("transformer.shared_head.output."):]
                mapped[f"{lp}.shared_head_head.{tail}"] = value
            elif suffix.startswith("shared_head.norm."):
                tail = suffix[len("shared_head.norm."):]
                mapped[f"{lp}.shared_head_norm.{tail}"] = value
            elif suffix.startswith(("shared_head.output.", "shared_head.head.")):
                tail = suffix.split("shared_head.", 1)[1].split(".", 1)[1]
                mapped[f"{lp}.shared_head_head.{tail}"] = value
            elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                mapped[f"{lp}.{suffix}"] = value
            else:
                mapped[f"{lp}.mtp_block.{suffix}"] = value
            break
    return mapped


def _apply_zero_centered_norm_shift(weights: dict[str, Any]) -> dict[str, Any]:
    """Add +1.0 to zero-centered (vanilla) norm weights.

    Idempotent: a centered weight has mean ~0 (<0.5); an already-shifted weight
    has mean ~1 (>=0.5) and is left untouched. Mirrors mlx-lm Step3p5 sanitize.
    """
    for key, value in list(weights.items()):
        if not str(key).endswith(".weight") or "norm" not in str(key):
            continue
        if getattr(value, "ndim", None) != 1:
            continue
        try:
            if float(value.mean().item()) < 0.5:
                weights[key] = value + 1.0
        except Exception:
            # Best-effort; non-MLX arrays are left as-is.
            pass
    return weights


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    start = int(
        text_config(config).get("num_hidden_layers")
        or config.get("num_hidden_layers")
        or 0
    )
    prefixes = tuple(
        tag
        for i in range(_num_mtp_layers(config))
        for tag in (
            f"model.layers.{start + i}.",
            f"language_model.model.layers.{start + i}.",
        )
    )
    return candidate_mtp_weight_files(model_path, config, prefixes)


def _load_raw_weights(paths: list[Path]) -> dict[str, Any]:
    import mlx.core as mx

    raw: dict[str, Any] = {}
    for path in paths:
        if path.suffix != ".safetensors":
            continue
        raw.update(dict(mx.load(str(path))))
    return raw


def _validate_mtp_load_coverage(mtp: Any, weights: dict[str, Any]) -> None:
    from mlx.utils import tree_flatten

    current = tree_flatten(mtp.parameters(), destination={})
    supplied = dict(weights)
    extra = sorted(set(supplied) - set(current))
    missing = sorted(set(current) - set(supplied))
    mismatched = [
        (key, tuple(current[key].shape), tuple(supplied[key].shape))
        for key in sorted(set(current) & set(supplied))
        if tuple(current[key].shape) != tuple(supplied[key].shape)
    ]
    if not extra and not missing and not mismatched:
        return
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing[:16]}" + (" ..." if len(missing) > 16 else ""))
    if extra:
        parts.append(f"extra={extra[:16]}" + (" ..." if len(extra) > 16 else ""))
    if mismatched:
        preview = [f"{k}: want {w}, got {g}" for k, w, g in mismatched[:8]]
        parts.append(f"shape_mismatch={preview}")
    raise ValueError("Step MTP overlay does not match runtime module tree: " + "; ".join(parts))


def _make_step_mtp_module(args: Any, num_mtp_layers: int, start_layer: int):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.step3p5 import Step3p5DecoderLayer, ZeroCenteredRMSNorm

    class _StepMTPLayer(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.enorm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hnorm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.mtp_block = Step3p5DecoderLayer(args, layer_idx=layer_idx)
            self.shared_head_norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.shared_head_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        def __call__(self, input_ids, previous_hidden_states, *, embed_tokens, cache=None):
            e = self.enorm(embed_tokens(input_ids))
            h = self.hnorm(previous_hidden_states)
            # vLLM concat order is [embedding, hidden].
            mixed = self.eh_proj(mx.concatenate([e, h], axis=-1))
            mask = create_attention_mask(mixed, cache)
            hidden = self.mtp_block(mixed, mask=mask, cache=cache)
            logits = self.shared_head_head(self.shared_head_norm(hidden))
            # The chained hidden for the next depth is the block output (pre
            # shared-head norm), matching the vLLM Step3p5MTP predictor.
            return logits, hidden

    class _StepMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                _StepMTPLayer(start_layer + idx) for idx in range(num_mtp_layers)
            ]
            self.start_layer = start_layer
            self.num_mtp_layers = num_mtp_layers

    return _StepMTP()


def _quantize_step_mtp_module(mtp: Any, contract: Any | None) -> None:
    if contract is None:
        return
    if getattr(contract, "mtp_prequantized", False):
        return
    if getattr(contract, "mtp_quant_bits", None) is None:
        return
    from .mtp_patch import _quantize_mtp_module

    _quantize_mtp_module(mtp, contract)


def inject_step3p5_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach Step-3.5/3.7 native MTP support to a loaded mlx-lm ``step3p5`` model."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_lm.models.step3p5 import ModelArgs

    if not is_step3p5_mtp_config(config):
        return False

    model_path = Path(model_path)
    tcfg = text_config(config)
    args = getattr(model, "args", None)
    if not isinstance(args, ModelArgs):
        args = ModelArgs.from_dict(tcfg)

    start_layer = int(getattr(args, "num_hidden_layers"))
    num_mtp_layers = _num_mtp_layers(config)
    hidden_variant = _hidden_variant_from_config(config, contract)
    sliding_window = int(getattr(args, "sliding_window", 512) or 512)

    raw = _load_raw_weights(_candidate_weight_files(model_path, config))
    raw = {k: v for k, v in raw.items() if _is_step_mtp_key(k, start_layer, num_mtp_layers)}
    mapped = _rewrite_step_mtp_weights(
        raw,
        start_layer=start_layer,
        num_mtp_layers=num_mtp_layers,
    )
    if not mapped:
        logger.warning("[Step MTP inject] No Step MTP weights found in %s", model_path)
        return False
    mapped = _apply_zero_centered_norm_shift(mapped)

    mtp = _make_step_mtp_module(args, num_mtp_layers, start_layer)
    _validate_mtp_load_coverage(mtp, mapped)
    mtp.load_weights(list(mapped.items()), strict=True)
    _quantize_step_mtp_module(mtp, contract)
    mx.eval(mtp.parameters())

    model.mtp = mtp
    model._mtplx_hidden_variant = hidden_variant
    model._mtplx_concat_order = "embedding_hidden"
    model._mtplx_mtp_quant_policy = getattr(contract, "mtp_quant_policy", None)

    original_class = model.__class__

    class _MTPLXStepModel(original_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            inner = self.model
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = inner.embed_tokens(inputs)
            if cache is None:
                cache = [None] * inner.num_layers

            full_mask = None
            swa_mask = None
            if inner._full_idx is not None:
                full_mask = create_attention_mask(h, cache[inner._full_idx])
            if inner._swa_idx is not None:
                swa_mask = create_attention_mask(
                    h, cache[inner._swa_idx], window_size=inner.args.sliding_window
                )
            for layer, layer_cache in zip(inner.layers, cache):
                mask = swa_mask if layer.is_sliding else full_mask
                h = layer(h, mask=mask, cache=layer_cache)

            pre_norm = h
            post_norm = inner.norm(h)
            logits = self.lm_head(post_norm)
            if not return_hidden:
                return logits
            variant = hidden_variant or getattr(self, "_mtplx_hidden_variant", "pre_norm")
            hidden = pre_norm if variant == "pre_norm" else post_norm
            return logits, hidden

        def _mtp_layer_cache(self, mtp_cache, depth):
            if mtp_cache is None:
                return None
            if isinstance(mtp_cache, list):
                return mtp_cache[depth]
            return mtp_cache

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "pre_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError("Step MTP backend supports embedding_hidden concat order only")
            depth = 0 if mtp_depth is None else max(int(mtp_depth) - 1, 0)
            depth %= len(self.mtp.layers)
            layer_cache = self._mtp_layer_cache(mtp_cache, depth)
            logits, hidden = self.mtp.layers[depth](
                next_token_ids,
                hidden_states,
                embed_tokens=self.model.embed_tokens,
                cache=layer_cache,
            )
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                mtp_depth=mtp_depth,
            )
            return hidden

        def _mtp_cache_factory(self):
            caches = []
            for layer in self.mtp.layers:
                if getattr(layer.mtp_block, "is_sliding", False):
                    caches.append(RotatingKVCache(max_size=sliding_window))
                else:
                    caches.append(KVCache())
            return caches

        def make_mtp_cache(self):
            return self._mtp_cache_factory()

        def make_cache(self):
            make_cache = getattr(super(), "make_cache", None)
            if callable(make_cache):
                return make_cache()
            return [KVCache() for _ in self.layers]

    model.__class__ = _MTPLXStepModel
    logger.info(
        "[Step MTP inject] Loaded %d tensors (%d layers, hidden_variant=%s) from %s",
        len(mapped),
        num_mtp_layers,
        hidden_variant,
        model_path,
    )
    return True


def _is_step_mtp_key(key: str, start_layer: int, num_mtp_layers: int) -> bool:
    k = _strip_outer_prefix(str(key))
    if k.startswith("mtp."):
        return True
    for i in range(num_mtp_layers):
        if k.startswith(f"model.layers.{start_layer + i}.") or k.startswith(
            f"layers.{start_layer + i}."
        ):
            return True
    return False


def validate_step_mtp_support(model: Any) -> bool:
    if getattr(model, "mtp", None) is None:
        return False
    if not getattr(model.mtp, "layers", None):
        return False
    return callable(getattr(model, "mtp_forward", None)) and callable(
        getattr(model, "make_mtp_cache", None)
    )
