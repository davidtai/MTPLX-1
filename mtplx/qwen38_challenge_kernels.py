"""Retained target-shaped kernels from the pinned Qwen 3.8 challenge."""

from __future__ import annotations

from typing import Any

_DUAL_RMS_CONCAT_KERNEL = None
_Q8_EMBED_DUAL_RMS_CONCAT_KERNEL = None
_QK_RMS_ROPE_KERNEL = None
qwen38_dual_norm_calls = 0
qwen38_q8_embed_dual_norm_calls = 0
qwen38_qk_rms_rope_calls = 0
qwen38_row24_qk_length_fallback_calls = 0
qwen38_row26_qk_widen_calls = 0
qwen38_row24_eval_ladder_calls = 0
qwen38_row26_prefill_ladder_calls = 0
_QWEN38_ATTENTION_ORIGINAL_CALL = None
_QWEN38_DFLASH_GQA_ORIGINAL = None
_QWEN38_DFLASH_GQA_ROUTED = None
_QWEN38_DFLASH_GQA_WIDTHS: tuple[int, ...] = ()


def qwen38_row24_async_eval(value: Any, *, row26: bool = False) -> None:
    import mlx.core as mx

    global qwen38_row24_eval_ladder_calls, qwen38_row26_prefill_ladder_calls
    qwen38_row24_eval_ladder_calls += 1
    if row26:
        qwen38_row26_prefill_ladder_calls += 1
    mx.async_eval(value)


def qwen38_row24_eval_ladder_counter_snapshot() -> int:
    return int(qwen38_row24_eval_ladder_calls)


def qwen38_row26_prefill_ladder_counter_snapshot() -> int:
    return int(qwen38_row26_prefill_ladder_calls)


def configure_qwen38_dflash_row24_eval_ladder(
    model: Any,
    *,
    active: bool,
    prefill_stride: int = 4,
) -> dict[str, int]:
    """Install the retained target-evaluation ladder in DFlash hidden capture."""

    if prefill_stride not in (3, 4):
        raise ValueError("DFlash evaluation ladder stride must be 3 or 4")
    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    if not active:
        if hasattr(inner, "_dflash_post_layer"):
            delattr(inner, "_dflash_post_layer")
        return {"active": 0, "prefill_stride": 0}

    decode_rungs = frozenset({0, 1, 9, 19, 29, 39, 49, 57})

    def post_layer(hidden_states: Any, layer_index: int) -> None:
        length = int(hidden_states.shape[1])
        row24_prefill = length >= 512
        row24_decode = length <= 9
        if (
            row24_prefill
            and (
                layer_index == 0
                or layer_index % prefill_stride == prefill_stride - 1
            )
        ) or (row24_decode and layer_index in decode_rungs):
            qwen38_row24_async_eval(
                hidden_states,
                row26=bool(prefill_stride == 3 and row24_prefill),
            )

    inner._dflash_post_layer = post_layer
    return {"active": 1, "prefill_stride": prefill_stride}


def configure_qwen38_dflash_gqa_widths(
    model: Any,
    *,
    active: bool,
    widths: tuple[int, ...] = (6, 7, 8),
) -> dict[str, Any]:
    """Route the measured 16K DFlash verify widths through per-head SDPA."""

    requested = tuple(int(width) for width in widths)
    if requested != (6, 7, 8):
        raise ValueError("Qwen 3.8 DFlash GQA widths must be exactly (6, 7, 8)")

    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    attention_modules = [
        attention
        for layer in list(getattr(inner, "layers", ()) or ())
        if (attention := getattr(layer, "self_attn", None)) is not None
    ]
    eligible = sum(_row21_attention_eligible(attention) for attention in attention_modules)
    if active and eligible != 16:
        raise ValueError(
            "Qwen 3.8 DFlash GQA route requires exactly 16 eligible attention modules"
        )

    from dflash_mlx.engine import target_qwen_gdn
    from dflash_mlx.engine.gqa_sdpa import (
        async_per_head_gqa_sdpa,
        per_head_gqa_sdpa,
        repeat_gqa_mask,
    )

    global _QWEN38_DFLASH_GQA_ORIGINAL
    global _QWEN38_DFLASH_GQA_ROUTED
    global _QWEN38_DFLASH_GQA_WIDTHS
    if _QWEN38_DFLASH_GQA_ORIGINAL is None:
        _QWEN38_DFLASH_GQA_ORIGINAL = target_qwen_gdn._gqa_reshape_sdpa

    if not active:
        if (
            _QWEN38_DFLASH_GQA_ROUTED is not None
            and target_qwen_gdn._gqa_reshape_sdpa is _QWEN38_DFLASH_GQA_ROUTED
        ):
            target_qwen_gdn._gqa_reshape_sdpa = _QWEN38_DFLASH_GQA_ORIGINAL
        _QWEN38_DFLASH_GQA_ROUTED = None
        _QWEN38_DFLASH_GQA_WIDTHS = ()
        return {
            "active": False,
            "eligible_modules": eligible,
            "widths": [],
        }

    if _QWEN38_DFLASH_GQA_ROUTED is not None:
        if target_qwen_gdn._gqa_reshape_sdpa is not _QWEN38_DFLASH_GQA_ROUTED:
            raise RuntimeError("Qwen 3.8 DFlash GQA route was replaced after installation")
        if _QWEN38_DFLASH_GQA_WIDTHS != requested:
            raise RuntimeError("Qwen 3.8 DFlash GQA route changed widths")
    else:
        original = _QWEN38_DFLASH_GQA_ORIGINAL

        def routed_gqa(
            queries: Any,
            keys: Any,
            values: Any,
            *,
            scale: float,
            mask: Any,
            cache: Any = None,
        ) -> Any:
            q_len = int(queries.shape[2])
            kv_len = int(keys.shape[2])
            if 16_384 <= kv_len < 32_768 and q_len in requested:
                grouped_mask = repeat_gqa_mask(
                    mask,
                    q_len=q_len,
                    kv_len=kv_len,
                    gqa=6,
                )
                if q_len < 8:
                    return async_per_head_gqa_sdpa(
                        queries,
                        keys,
                        values,
                        scale=scale,
                        mask=grouped_mask,
                        gqa=6,
                    )
                return per_head_gqa_sdpa(
                    queries,
                    keys,
                    values,
                    scale=scale,
                    mask=grouped_mask,
                    gqa=6,
                )
            return original(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                cache=cache,
            )

        _QWEN38_DFLASH_GQA_ROUTED = routed_gqa
        _QWEN38_DFLASH_GQA_WIDTHS = requested
        target_qwen_gdn._gqa_reshape_sdpa = routed_gqa

    return {
        "active": True,
        "eligible_modules": eligible,
        "widths": list(requested),
        "async_widths": [6, 7],
        "per_head_widths": [8],
        "kv_len_min": 16_384,
        "kv_len_max_exclusive": 32_768,
    }


def configure_qwen38_dflash_m8_nax_island(
    model: Any,
    *,
    active: bool,
    include_linear_z: bool = False,
    include_m7_output: bool = False,
    include_m7_linear_z: bool = False,
) -> dict[str, Any]:
    """Route only measured width-8 Qwen attention projections to BM=8 NAX."""

    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    attention_modules = [
        attention
        for layer in list(getattr(inner, "layers", ()) or ())
        if (attention := getattr(layer, "self_attn", None)) is not None
    ]
    eligible = [attention for attention in attention_modules if _row21_attention_eligible(attention)]
    projection_shapes: list[tuple[int, int]] = []
    for attention in eligible:
        for projection in (attention.q_proj, attention.o_proj):
            weight = projection["weight"]
            bits = int(getattr(projection, "bits", 0))
            group_size = int(getattr(projection, "group_size", 0))
            if bits != 4 or group_size != 32:
                raise ValueError("Qwen 3.8 M8 NAX island requires affine Q4/G32")
            projection_shapes.append(
                (int(weight.shape[1]) * 32 // bits, int(weight.shape[0]))
            )
    expected = [(5_120, 12_288), (6_144, 5_120)] * 16
    if len(eligible) != 16 or sorted(projection_shapes) != sorted(expected):
        raise ValueError(
            "Qwen 3.8 M8 NAX island projection geometry changed: "
            f"attention={len(attention_modules)}, eligible={len(eligible)}, "
            f"shapes={sorted(set(projection_shapes))}, "
            f"shape_counts={[(shape, projection_shapes.count(shape)) for shape in sorted(set(projection_shapes))]}"
        )

    linear_modules = [
        linear
        for layer in list(getattr(inner, "layers", ()) or ())
        if (linear := getattr(layer, "linear_attn", None)) is not None
    ]
    linear_z_shapes: list[tuple[int, int]] = []
    for linear in linear_modules:
        projection = linear.in_proj_z
        weight = projection["weight"]
        bits = int(getattr(projection, "bits", 0))
        group_size = int(getattr(projection, "group_size", 0))
        if bits != 4 or group_size != 32:
            raise ValueError("Qwen 3.8 M8 linear-Z island requires affine Q4/G32")
        linear_z_shapes.append(
            (int(weight.shape[1]) * 32 // bits, int(weight.shape[0]))
        )
    if include_linear_z and linear_z_shapes != [(5_120, 6_144)] * 48:
        raise ValueError("Qwen 3.8 M8 linear-Z projection geometry changed")

    from mtplx.nax_verify import configure_qwen38_m8_nax_island

    report = configure_qwen38_m8_nax_island(
        active=active,
        include_linear_z=include_linear_z,
        include_m7_output=include_m7_output,
        include_m7_linear_z=include_m7_linear_z,
    )
    routed_shapes = {tuple(shape) for shape in report["shapes"]}
    all_projection_shapes = projection_shapes + linear_z_shapes
    return {
        **report,
        "eligible_attention_modules": len(eligible),
        "eligible_linear_z_projections": sum(
            shape in routed_shapes for shape in linear_z_shapes
        ),
        "eligible_m7_projections": sum(
            shape in {tuple(item) for item in report["m7_shapes"]}
            for shape in all_projection_shapes
        ),
        "eligible_m7_linear_z_projections": (
            linear_z_shapes.count((5_120, 6_144))
            if include_m7_linear_z
            else 0
        ),
        "validated_projections": len(projection_shapes),
        "eligible_projections": sum(
            shape in routed_shapes for shape in all_projection_shapes
        ),
    }


def qwen38_qk_rms_rope(
    queries: Any,
    keys: Any,
    q_weight: Any,
    k_weight: Any,
    eps: float,
    offset: int,
) -> tuple[Any, Any]:
    """Fuse exact Qwen 3.8 BF16 Q/K RMSNorm, transpose, and partial RoPE."""

    import mlx.core as mx

    global _QK_RMS_ROPE_KERNEL, qwen38_qk_rms_rope_calls
    if (
        queries.ndim != 4
        or keys.ndim != 4
        or tuple(queries.shape[0:2]) != tuple(keys.shape[0:2])
        or tuple(queries.shape[2:]) != (24, 256)
        or tuple(keys.shape[2:]) != (4, 256)
        or queries.dtype != mx.bfloat16
        or keys.dtype != mx.bfloat16
        or tuple(q_weight.shape) != (256,)
        or tuple(k_weight.shape) != (256,)
        or q_weight.dtype != mx.bfloat16
        or k_weight.dtype != mx.bfloat16
    ):
        raise ValueError("unsupported Qwen 3.8 Q/K RMSNorm+RoPE contract")
    if _QK_RMS_ROPE_KERNEL is None:
        _QK_RMS_ROPE_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_qk_rms_rope_bf16_h256_r64_v1",
            input_names=[
                "q", "k", "q_weight", "k_weight", "eps", "offset", "log2_base"
            ],
            output_names=["q_out", "k_out"],
            source=r"""
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint rotary_dimensions = 64;
                constexpr uint rotary_pairs = rotary_dimensions / 2;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;

                uint batch_size = uint(q_shape[0]);
                uint sequence_length = uint(q_shape[1]);
                uint query_heads = uint(q_shape[2]);
                uint key_heads = uint(k_shape[2]);
                uint axis_size = uint(q_shape[3]);
                uint query_rows = batch_size * query_heads * sequence_length;
                bool is_query = row < query_rows;
                uint local_row = is_query ? row : row - query_rows;
                uint head_count = is_query ? query_heads : key_heads;
                uint batch = local_row / (head_count * sequence_length);
                uint head_sequence = local_row % (head_count * sequence_length);
                uint head = head_sequence / sequence_length;
                uint sequence = head_sequence % sequence_length;

                ulong input_base;
                ulong input_axis_stride;
                ulong output_base = ulong(local_row) * ulong(axis_size);
                if (is_query) {
                    input_base = ulong(batch) * ulong(q_strides[0])
                        + ulong(sequence) * ulong(q_strides[1])
                        + ulong(head) * ulong(q_strides[2]);
                    input_axis_stride = ulong(q_strides[3]);
                } else {
                    input_base = ulong(batch) * ulong(k_strides[0])
                        + ulong(sequence) * ulong(k_strides[1])
                        + ulong(head) * ulong(k_strides[2]);
                    input_axis_stride = ulong(k_strides[3]);
                }

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                threadgroup bfloat normalized[256];

                float acc = 0.0f;
                uint first = thread_id * n_reads;
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element < axis_size) {
                        ulong index = input_base + ulong(element) * input_axis_stride;
                        float value = is_query ? float(q[index]) : float(k[index]);
                        acc += value * value;
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) local_sums[simd_thread] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) local_sums[simd_group] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / axis_size + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element < axis_size) {
                        ulong index = input_base + ulong(element) * input_axis_stride;
                        bfloat input_value = is_query ? q[index] : k[index];
                        bfloat rms_value = bfloat(float(input_value) * inv_mean);
                        bfloat weight = is_query
                            ? q_weight[element] : k_weight[element];
                        normalized[element] = weight * rms_value;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0; i < n_reads; ++i) {
                    uint element = first + i;
                    if (element >= rotary_dimensions && element < axis_size) {
                        if (is_query) q_out[output_base + element] = normalized[element];
                        else k_out[output_base + element] = normalized[element];
                    }
                }
                if (thread_id < rotary_pairs / n_reads) {
                    for (uint i = 0; i < n_reads; ++i) {
                        uint pair = first + i;
                        float d = float(pair) / float(rotary_pairs);
                        float inv_freq = metal::exp2(-d * float(log2_base));
                        float position = float(int(sequence) + int(offset));
                        float theta = position * inv_freq;
                        float costheta = metal::fast::cos(theta);
                        float sintheta = metal::fast::sin(theta);
                        float x1 = float(normalized[pair]);
                        float x2 = float(normalized[pair + rotary_pairs]);
                        bfloat rx1 = bfloat(x1 * costheta - x2 * sintheta);
                        bfloat rx2 = bfloat(x1 * sintheta + x2 * costheta);
                        if (is_query) {
                            q_out[output_base + pair] = rx1;
                            q_out[output_base + pair + rotary_pairs] = rx2;
                        } else {
                            k_out[output_base + pair] = rx1;
                            k_out[output_base + pair + rotary_pairs] = rx2;
                        }
                    }
                }
            """,
            ensure_row_contiguous=False,
        )
    batch, length = int(queries.shape[0]), int(queries.shape[1])
    qwen38_qk_rms_rope_calls += 1
    q_out, k_out = _QK_RMS_ROPE_KERNEL(
        inputs=[
            queries,
            keys,
            q_weight,
            k_weight,
            float(eps),
            int(offset),
            23.253496664211536,
        ],
        template=[],
        grid=(batch * length * (24 + 4) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(batch, 24, length, 256), (batch, 4, length, 256)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return q_out, k_out


def qwen38_qk_rms_rope_counter_snapshot() -> int:
    return int(qwen38_qk_rms_rope_calls)


def _row21_attention_eligible(attention: Any) -> bool:
    rope = getattr(attention, "rope", None)
    return bool(
        int(getattr(attention, "num_attention_heads", 0)) == 24
        and int(getattr(attention, "num_key_value_heads", 0)) == 4
        and int(getattr(attention, "head_dim", 0)) == 256
        and rope is not None
        and int(getattr(rope, "dims", 0)) == 64
        and float(getattr(rope, "base", 0.0)) == 10_000_000.0
        and float(getattr(rope, "scale", 0.0)) == 1.0
        and not bool(getattr(rope, "traditional", True))
        and float(attention.q_norm.eps) == float(attention.k_norm.eps)
    )


def configure_qwen38_row21_qk_rms_rope(model: Any, *, active: bool) -> dict[str, int]:
    """Toggle the row-21 fused Q/K preparation on exact Qwen 3.8 attention."""

    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.qwen3_5 import Attention

    global _QWEN38_ATTENTION_ORIGINAL_CALL
    if _QWEN38_ATTENTION_ORIGINAL_CALL is None:
        _QWEN38_ATTENTION_ORIGINAL_CALL = Attention.__call__

        def row21_attention_call(self, x, mask=None, cache=None):
            if not bool(getattr(self, "_mtplx_qwen38_row21_active", False)):
                return _QWEN38_ATTENTION_ORIGINAL_CALL(
                    self,
                    x,
                    mask=mask,
                    cache=cache,
                )
            if not _row21_attention_eligible(self):
                raise ValueError("active Qwen 3.8 row-21 attention is ineligible")
            max_length = getattr(self, "_mtplx_qwen38_row24_qk_max_length", None)
            if max_length is not None and int(x.shape[1]) > int(max_length):
                global qwen38_row24_qk_length_fallback_calls
                qwen38_row24_qk_length_fallback_calls += 1
                return _QWEN38_ATTENTION_ORIGINAL_CALL(
                    self,
                    x,
                    mask=mask,
                    cache=cache,
                )
            if max_length == 32 and 16 < int(x.shape[1]) <= 32:
                global qwen38_row26_qk_widen_calls
                qwen38_row26_qk_widen_calls += 1
            offset = getattr(cache, "offset", 0) if cache is not None else 0
            if isinstance(offset, mx.array):
                # Matches the source patch's `hasArrayOffset` fallback: the
                # device draft core keeps this offset traced, while the fused
                # Metal kernel takes a host scalar.
                return _QWEN38_ATTENTION_ORIGINAL_CALL(
                    self,
                    x,
                    mask=mask,
                    cache=cache,
                )
            batch, length, _ = x.shape
            q_projection = self.q_proj(x)
            queries, gate = mx.split(
                q_projection.reshape(batch, length, 24, -1),
                2,
                axis=-1,
            )
            keys = self.k_proj(x).reshape(batch, length, 4, 256)
            values = self.v_proj(x).reshape(batch, length, 4, 256).transpose(
                0, 2, 1, 3
            )
            queries, keys = qwen38_qk_rms_rope(
                queries,
                keys,
                self.q_norm.weight,
                self.k_norm.weight,
                float(self.q_norm.eps),
                int(offset),
            )
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return self.o_proj(output * mx.sigmoid(gate.reshape(batch, length, -1)))

        Attention.__call__ = row21_attention_call

    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    layers = list(getattr(inner, "layers", ()) or ())
    eligible = 0
    dflash_modules = 0
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        is_eligible = _row21_attention_eligible(attention)
        attention._mtplx_qwen38_row21_active = bool(active and is_eligible)
        if active and is_eligible:
            def dflash_qk_prepare(
                queries,
                keys,
                offset,
                *,
                _attention=attention,
            ):
                return qwen38_qk_rms_rope(
                    queries,
                    keys,
                    _attention.q_norm.weight,
                    _attention.k_norm.weight,
                    float(_attention.q_norm.eps),
                    int(offset),
                )

            attention._dflash_qk_prepare = dflash_qk_prepare
            dflash_modules += 1
        elif hasattr(attention, "_dflash_qk_prepare"):
            delattr(attention, "_dflash_qk_prepare")
        eligible += int(is_eligible)
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "dflash_modules": dflash_modules,
        "mtp_array_offset_skipped": 1,
    }


def configure_qwen38_row24_qk_length_limit(
    model: Any,
    *,
    active: bool,
    max_length: int = 16,
) -> dict[str, int]:
    """Apply row 24's L<=16 bound to the retained row-21 fusion."""

    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    eligible = 0
    dflash_modules = 0
    for layer in list(getattr(inner, "layers", ()) or ()):
        attention = getattr(layer, "self_attn", None)
        if attention is None or not _row21_attention_eligible(attention):
            continue
        attention._mtplx_qwen38_row24_qk_max_length = max_length if active else None
        if active:
            def dflash_qk_fallback() -> None:
                global qwen38_row24_qk_length_fallback_calls
                qwen38_row24_qk_length_fallback_calls += 1

            attention._dflash_qk_max_length = max_length
            attention._dflash_qk_fallback = dflash_qk_fallback
            dflash_modules += 1
        else:
            for attr in ("_dflash_qk_max_length", "_dflash_qk_fallback"):
                if hasattr(attention, attr):
                    delattr(attention, attr)
        eligible += 1
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "dflash_modules": dflash_modules,
        "max_length": max_length if active else 0,
    }


def qwen38_row24_qk_length_fallback_counter_snapshot() -> int:
    return int(qwen38_row24_qk_length_fallback_calls)


def qwen38_row26_qk_widen_counter_snapshot() -> int:
    return int(qwen38_row26_qk_widen_calls)


def qwen38_dual_rms_norm_concat(
    a: Any,
    b: Any,
    a_weight: Any,
    b_weight: Any,
    eps: float,
) -> Any:
    """Normalize two hidden-5120 BF16 rows and emit one contiguous concat."""

    import mlx.core as mx

    global _DUAL_RMS_CONCAT_KERNEL, qwen38_dual_norm_calls
    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != mx.bfloat16
        or b.dtype != mx.bfloat16
        or int(a.shape[-1]) != 5120
        or tuple(a_weight.shape) != (5120,)
        or tuple(b_weight.shape) != (5120,)
        or a_weight.dtype != mx.bfloat16
        or b_weight.dtype != mx.bfloat16
    ):
        raise ValueError("unsupported Qwen 3.8 dual RMSNorm contract")
    rows = int(a.size) // 5120
    if _DUAL_RMS_CONCAT_KERNEL is None:
        _DUAL_RMS_CONCAT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_dual_rms_norm_concat_bf16_v1",
            input_names=["a", "b", "a_weight", "b_weight", "eps"],
            output_names=["concat_out"],
            source=r"""
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint lsize = 1024;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint axis_size = uint(a_shape[a_ndim - 1]);
                uint a_rows = 1;
                for (uint i = 0; i + 1 < a_ndim; ++i) {
                    a_rows *= uint(a_shape[i]);
                }
                bool is_a = row < a_rows;
                uint local_row = is_a ? row : row - a_rows;
                ulong in_off = ulong(local_row) * ulong(axis_size);
                ulong out_off = ulong(local_row) * ulong(axis_size * 2)
                    + (is_a ? 0 : ulong(axis_size));

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0.0f;
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) {
                    local_sums[simd_thread] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) {
                    local_sums[simd_group] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / float(axis_size) + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            bfloat wi = is_a
                                ? a_weight[elem + i] : b_weight[elem + i];
                            concat_out[out_off + elem + i] =
                                wi * bfloat(xi * inv_mean);
                        }
                    }
                }
            """,
            ensure_row_contiguous=True,
        )
    qwen38_dual_norm_calls += 1
    (output,) = _DUAL_RMS_CONCAT_KERNEL(
        inputs=[a, b, a_weight, b_weight, float(eps)],
        template=[],
        grid=(2 * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*a.shape[:-1], 10240)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def reset_qwen38_dual_norm_calls() -> None:
    global qwen38_dual_norm_calls

    qwen38_dual_norm_calls = 0


def qwen38_dual_norm_counter_snapshot() -> int:
    return int(qwen38_dual_norm_calls)


def qwen38_q8_embedding_dual_rms_norm_concat(
    token_ids: Any,
    embedding: Any,
    hidden: Any,
    embedding_norm_weight: Any,
    hidden_norm_weight: Any,
    eps: float,
) -> Any:
    """Fuse the target's Q8/g64 embedding lookup with both MTP input norms."""

    import mlx.core as mx

    global _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL, qwen38_q8_embed_dual_norm_calls
    weight = getattr(embedding, "weight", None)
    scales = getattr(embedding, "scales", None)
    biases = getattr(embedding, "biases", None)
    rows = int(hidden.size) // 5120
    if (
        int(getattr(embedding, "bits", 0)) != 8
        or int(getattr(embedding, "group_size", 0)) != 64
        or str(getattr(embedding, "mode", "")).lower() != "affine"
        or weight is None
        or scales is None
        or biases is None
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or hidden.dtype != mx.bfloat16
        or int(hidden.shape[-1]) != 5120
        or int(token_ids.size) != rows
        or tuple(weight.shape[1:]) != (1280,)
        or tuple(scales.shape[1:]) != (80,)
        or tuple(biases.shape[1:]) != (80,)
        or tuple(embedding_norm_weight.shape) != (5120,)
        or tuple(hidden_norm_weight.shape) != (5120,)
        or embedding_norm_weight.dtype != mx.bfloat16
        or hidden_norm_weight.dtype != mx.bfloat16
    ):
        raise ValueError("unsupported Qwen 3.8 Q8 embedding dual RMSNorm contract")

    if _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL is None:
        _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_q8_embedding_dual_rms_norm_concat_bf16_v1",
            input_names=[
                "token_ids",
                "embedding_weight",
                "embedding_scales",
                "embedding_biases",
                "hidden",
                "embedding_norm_weight",
                "hidden_norm_weight",
                "eps",
            ],
            output_names=["concat_out"],
            source=r"""
                constexpr uint axis_size = 5120;
                constexpr uint group_size = 64;
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint lsize = 1024;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint hidden_rows = 1;
                for (uint i = 0; i + 1 < hidden_ndim; ++i) {
                    hidden_rows *= uint(hidden_shape[i]);
                }
                bool is_embedding = row < hidden_rows;
                uint local_row = is_embedding ? row : row - hidden_rows;
                long token = is_embedding ? long(token_ids[local_row]) : 0;
                ulong packed_off = ulong(token) * ulong(embedding_weight_shape[1]);
                ulong scale_off = ulong(token) * ulong(embedding_scales_shape[1]);
                ulong hidden_off = ulong(local_row) * ulong(axis_size);
                ulong out_off = hidden_off * 2ul
                    + (is_embedding ? 0ul : ulong(axis_size));

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0.0f;
                for (uint start = 0; start < axis_size; start += lsize * n_reads) {
                    uint elem = start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        uint index = elem + i;
                        if (index < axis_size) {
                            float xi;
                            if (is_embedding) {
                                uint packed = embedding_weight[
                                    packed_off + ulong(index >> 2)];
                                uint q = (packed >> ((index & 3u) * 8u)) & 255u;
                                bfloat dequantized = bfloat(q)
                                    * embedding_scales[
                                        scale_off + ulong(index / group_size)]
                                    + embedding_biases[
                                        scale_off + ulong(index / group_size)];
                                xi = float(dequantized);
                            } else {
                                xi = float(hidden[hidden_off + ulong(index)]);
                            }
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) local_sums[simd_thread] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) local_sums[simd_group] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / float(axis_size) + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint start = 0; start < axis_size; start += lsize * n_reads) {
                    uint elem = start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        uint index = elem + i;
                        if (index < axis_size) {
                            float xi;
                            if (is_embedding) {
                                uint packed = embedding_weight[
                                    packed_off + ulong(index >> 2)];
                                uint q = (packed >> ((index & 3u) * 8u)) & 255u;
                                bfloat dequantized = bfloat(q)
                                    * embedding_scales[
                                        scale_off + ulong(index / group_size)]
                                    + embedding_biases[
                                        scale_off + ulong(index / group_size)];
                                xi = float(dequantized);
                            } else {
                                xi = float(hidden[hidden_off + ulong(index)]);
                            }
                            bfloat wi = is_embedding
                                ? embedding_norm_weight[index]
                                : hidden_norm_weight[index];
                            concat_out[out_off + ulong(index)] =
                                wi * bfloat(xi * inv_mean);
                        }
                    }
                }
            """,
            ensure_row_contiguous=False,
        )
    qwen38_q8_embed_dual_norm_calls += 1
    (output,) = _Q8_EMBED_DUAL_RMS_CONCAT_KERNEL(
        inputs=[
            token_ids.reshape(-1),
            weight,
            scales,
            biases,
            hidden.reshape(rows, 5120),
            embedding_norm_weight,
            hidden_norm_weight,
            float(eps),
        ],
        template=[],
        grid=(2 * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*hidden.shape[:-1], 10240)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def qwen38_q8_embedding_dual_norm_counter_snapshot() -> int:
    return int(qwen38_q8_embed_dual_norm_calls)
