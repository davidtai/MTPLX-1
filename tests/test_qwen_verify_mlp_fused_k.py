"""Experimental M2-M6 Qwen gate/up fusion on the measured vk_k geometry."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mtplx.kernels.qwen_verify_mlp_fused_k import (
    _fused_gate_up_q4_ksplit_source,
    is_qwen_gate_up_swiglu_vk_k_eligible,
    qwen_gate_up_swiglu_vk_k,
)


def _q4_linear(k: int, n: int, *, group_size: int = 64) -> nn.QuantizedLinear:
    module = nn.QuantizedLinear(
        k,
        n,
        bias=False,
        group_size=group_size,
        bits=4,
        mode="affine",
    )
    module.scales = module.scales.astype(mx.bfloat16)
    module.biases = module.biases.astype(mx.bfloat16)
    return module


def test_source_preserves_vk_k_ownership_and_projection_rounding() -> None:
    source = _fused_gate_up_q4_ksplit_source(
        m=4,
        group_size=64,
        k_parts=2,
        kconst=5120,
    )

    # Same output-tile-owned exact-M4 split-K geometry as the current winner.
    assert "constexpr int K = 5120;" in source
    assert "constexpr int K_PARTS = 2;" in source
    assert "int n0 = int(tg_n) * 4;" in source
    assert "float gate_acc[16];" in source
    assert "float up_acc[16];" in source
    assert "threadgroup float gate_partials[K_PARTS * 16];" in source
    assert "threadgroup float up_partials[K_PARTS * 16];" in source
    assert "for (int p = 0; p < K_PARTS; ++p)" in source

    # The two original Q4 arrays share each activation load; no packed copy.
    assert "gate_w_q" in source
    assert "up_w_q" in source
    assert "packed_gateup" not in source
    for row in range(4):
        assert source.count(f"Vec8 vA_{row} =") == 1

    # Match stock's projection boundary: round each projection to T before
    # precise SwiGLU. This changes the FP32 reduction tree, not its precision.
    assert "T gate_value = T(gate_total);" in source
    assert "T up_value = T(up_total);" in source
    assert "metal::exp(metal::abs(gate_value))" in source
    assert "metal::fast" not in source
    assert "atomic" not in source


def test_bn2_source_stays_below_the_proven_accumulator_ceiling() -> None:
    source = _fused_gate_up_q4_ksplit_source(
        m=4,
        group_size=64,
        k_parts=2,
        kconst=5120,
        tile_cols=2,
    )

    assert "int n0 = int(tg_n) * 2;" in source
    assert "float gate_acc[8];" in source
    assert "float up_acc[8];" in source
    assert "threadgroup float gate_partials[K_PARTS * 8];" in source
    assert "threadgroup float up_partials[K_PARTS * 8];" in source
    assert "n0 + 2" not in source

    source_m6 = _fused_gate_up_q4_ksplit_source(
        m=6,
        group_size=64,
        k_parts=2,
        kconst=5120,
        tile_cols=2,
    )
    assert "float gate_acc[12];" in source_m6
    assert "float up_acc[12];" in source_m6
    assert "threadgroup float gate_partials[K_PARTS * 12];" in source_m6
    assert "threadgroup float up_partials[K_PARTS * 12];" in source_m6


def test_exact_m_source_covers_every_verify_tile() -> None:
    # M2/M3/M5 compile as their own exact tiles instead of padding into M4/M6,
    # so no zero-padded activation copy is materialized on those calls.
    for m, tile_cols in ((2, 2), (3, 2), (5, 2), (3, 4), (5, 4)):
        source = _fused_gate_up_q4_ksplit_source(
            m=m,
            group_size=64,
            k_parts=2,
            kconst=5120,
            tile_cols=tile_cols,
        )
        n_acc = tile_cols * m
        assert f"float gate_acc[{n_acc}];" in source
        assert f"float up_acc[{n_acc}];" in source
        assert f"threadgroup float gate_partials[K_PARTS * {n_acc}];" in source
        assert f"int n0 = int(tg_n) * {tile_cols};" in source
        # BN=2 M6 is the documented 24-accumulator ceiling; nothing exceeds it.
        assert n_acc <= 24


def test_eligibility_is_exact_m4_q4_affine_no_bias() -> None:
    k, n = 512, 4096
    gate = _q4_linear(k, n)
    up = _q4_linear(k, n)
    x4 = mx.zeros((4, k), dtype=mx.bfloat16)

    for rows in range(2, 7):
        assert is_qwen_gate_up_swiglu_vk_k_eligible(
            mx.zeros((rows, k), dtype=mx.bfloat16),
            gate,
            up,
        )
    assert not is_qwen_gate_up_swiglu_vk_k_eligible(x4[:1], gate, up)
    assert not is_qwen_gate_up_swiglu_vk_k_eligible(
        mx.zeros((7, k), dtype=mx.bfloat16),
        gate,
        up,
    )
    assert not is_qwen_gate_up_swiglu_vk_k_eligible(
        x4,
        gate,
        _q4_linear(k, n + 4),
    )
    assert not is_qwen_gate_up_swiglu_vk_k_eligible(
        x4,
        gate,
        nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8),
    )


def test_fused_vk_k_matches_two_current_vk_k_projections() -> None:
    from mlx_lm.models.qwen3_next import swiglu

    from mtplx.verify_kernels import vk_qmm_m4_impl

    k, n = 512, 4096
    mx.random.seed(17)
    gate = _q4_linear(k, n)
    up = _q4_linear(k, n)
    x = (mx.random.normal((4, k), dtype=mx.float32) * 0.25).astype(mx.bfloat16)
    assert is_qwen_gate_up_swiglu_vk_k_eligible(x, gate, up)

    reference = swiglu(
        vk_qmm_m4_impl(
            "vk_k",
            x,
            gate.weight,
            gate.scales,
            gate.biases,
            bits=4,
            group_size=64,
        ),
        vk_qmm_m4_impl(
            "vk_k",
            x,
            up.weight,
            up.scales,
            up.biases,
            bits=4,
            group_size=64,
        ),
    )
    for tile_cols in (2, 4):
        candidate = qwen_gate_up_swiglu_vk_k(
            x,
            gate,
            up,
            tile_cols=tile_cols,
        )
        replay = qwen_gate_up_swiglu_vk_k(
            x,
            gate,
            up,
            tile_cols=tile_cols,
        )
        mx.eval(reference, candidate, replay)

        dmax = float(
            mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32)).max()
        )
        replay_dmax = float(
            mx.abs(candidate.astype(mx.float32) - replay.astype(mx.float32)).max()
        )
        assert candidate.shape == (4, n)
        assert dmax <= 0.25, f"fused BN={tile_cols} drift too large: {dmax}"
        assert replay_dmax == 0.0


def test_fused_bn2_matches_stock_for_every_speculative_verify_shape() -> None:
    from mlx_lm.models.qwen3_next import swiglu

    k, n = 512, 4096
    mx.random.seed(23)
    gate = _q4_linear(k, n)
    up = _q4_linear(k, n)

    for rows in range(2, 7):
        x = (mx.random.normal((rows, k), dtype=mx.float32) * 0.25).astype(mx.bfloat16)
        assert is_qwen_gate_up_swiglu_vk_k_eligible(x, gate, up)
        reference = swiglu(gate(x), up(x))
        candidate = qwen_gate_up_swiglu_vk_k(
            x,
            gate,
            up,
            tile_cols=2,
        )
        mx.eval(reference, candidate)
        dmax = float(
            mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32)).max()
        )
        assert candidate.shape == (rows, n)
        assert dmax <= 0.25, f"M={rows} fused BN=2 drift too large: {dmax}"


def test_exact_tile_is_bit_identical_to_padded_tile() -> None:
    # Removing the zero-pad is a pure dispatch/compute-waste optimization: the
    # exact M2/M3/M5 tile must reproduce the padded M4/M6 tile bit-for-bit on
    # the live rows, so token trajectories are unchanged from the padded path.
    k, n = 512, 4096
    mx.random.seed(29)
    gate = _q4_linear(k, n)
    up = _q4_linear(k, n)

    for rows, padded in ((2, 4), (3, 4), (5, 6)):
        x = (mx.random.normal((rows, k), dtype=mx.float32) * 0.25).astype(mx.bfloat16)
        pad = mx.concatenate(
            [x, mx.zeros((padded - rows, k), dtype=mx.bfloat16)],
            axis=0,
        )
        for tile_cols in (2, 4):
            exact = qwen_gate_up_swiglu_vk_k(x, gate, up, tile_cols=tile_cols)
            padded_rows = qwen_gate_up_swiglu_vk_k(
                pad, gate, up, tile_cols=tile_cols
            )[:rows, :]
            mx.eval(exact, padded_rows)
            drift = float(
                mx.abs(
                    exact.astype(mx.float32) - padded_rows.astype(mx.float32)
                ).max()
            )
            assert exact.shape == (rows, n)
            assert drift == 0.0, f"M{rows} BN={tile_cols} exact!=padded: {drift}"


def test_internal_native_mlp_selector_accepts_fused_vk_k(monkeypatch) -> None:
    from mtplx import native_mlp

    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "fused-gateup-vk-k")
    assert native_mlp._normalized_variant() == "fused_gateup_vk_k"
    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "fused-gateup-vk-k-bn2")
    assert native_mlp._normalized_variant() == "fused_gateup_vk_k_bn2"
