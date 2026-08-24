"""Tests for the m4/NAX verify kernel module."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.nax_verify import (
    install_nax_qlinear_patch,
    m4_ksplit_eligible,
    m16_nax_eligible,
    nax_dispatch_counter_snapshot,
    nax_available,
    nax_qmm_m4,
    nax_qmm_m16,
    uninstall_nax_qlinear_patch,
)


def test_m8_nax_route_counter_records_exact_shape(monkeypatch) -> None:
    from mtplx import nax_verify

    class FakeKernel:
        def __call__(self, *, output_shapes, output_dtypes, **_kwargs):
            return [mx.zeros(output_shapes[0], dtype=output_dtypes[0])]

    monkeypatch.setattr(
        nax_verify,
        "_build_kernel_m8_nax_ktmpl",
        lambda *_args, **_kwargs: FakeKernel(),
    )
    before = nax_dispatch_counter_snapshot()
    x = mx.zeros((8, 32), dtype=mx.bfloat16)
    w_q = mx.zeros((64, 4), dtype=mx.uint32)
    scales = mx.zeros((64, 1), dtype=mx.bfloat16)
    biases = mx.zeros((64, 1), dtype=mx.bfloat16)
    y = nax_verify.nax_qmm_m8_nax(
        x,
        w_q,
        scales,
        biases,
        group_size=32,
    )
    assert y.shape == (8, 64)
    after = nax_dispatch_counter_snapshot()
    assert after["m8_nax"] == before.get("m8_nax", 0) + 1
    assert after["m8_nax_k32_n64"] == before.get("m8_nax_k32_n64", 0) + 1


def test_m8_nax_allows_shape_screening_with_explicit_nsg(monkeypatch) -> None:
    from mtplx import nax_verify

    observed = {}

    class FakeKernel:
        def __call__(self, *, grid, threadgroup, output_shapes, output_dtypes, **_kwargs):
            observed["grid"] = grid
            observed["threadgroup"] = threadgroup
            return [mx.zeros(output_shapes[0], dtype=output_dtypes[0])]

    def build(k, group_size, dtype, *, nsg=8):
        observed["build"] = (k, group_size, dtype, nsg)
        return FakeKernel()

    monkeypatch.setattr(nax_verify, "_build_kernel_m8_nax_ktmpl", build)
    y = nax_verify.nax_qmm_m8_nax(
        mx.zeros((8, 32), dtype=mx.bfloat16),
        mx.zeros((64, 4), dtype=mx.uint32),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        group_size=32,
        nsg=4,
    )
    assert y.shape == (8, 64)
    assert observed["build"] == (32, 32, mx.bfloat16, 4)
    assert observed["grid"] == (128, 2, 1)
    assert observed["threadgroup"] == (128, 1, 1)


def test_m7_route_pads_to_m8_tile_and_records_original_width(monkeypatch) -> None:
    from mtplx import nax_verify

    class FakeKernel:
        def __call__(self, *, output_shapes, output_dtypes, **_kwargs):
            assert output_shapes == [(8, 64)]
            return [mx.zeros(output_shapes[0], dtype=output_dtypes[0])]

    monkeypatch.setattr(
        nax_verify,
        "_build_kernel_m8_nax_ktmpl",
        lambda *_args, **_kwargs: FakeKernel(),
    )
    before = nax_dispatch_counter_snapshot()
    y = nax_verify.nax_qmm_m8_nax(
        mx.zeros((7, 32), dtype=mx.bfloat16),
        mx.zeros((64, 4), dtype=mx.uint32),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        group_size=32,
    )
    assert y.shape == (7, 64)
    after = nax_dispatch_counter_snapshot()
    assert after["m7_to_m8_nax"] == before.get("m7_to_m8_nax", 0) + 1
    assert after["m7_to_m8_nax_k32_n64"] == (
        before.get("m7_to_m8_nax_k32_n64", 0) + 1
    )


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_m7_padded_m8_nax_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import nax_qmm_m8_nax

    k, n = 512, 256
    w_q, scales, biases = _quantized_fixture(k, n)
    x = (mx.random.normal((7, k), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m8_nax(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    mx.eval(y, ref)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (7, n)
    assert diff < 0.25


def test_eligibility_shape_policy() -> None:
    dt = mx.bfloat16
    # m4: exact 4 rows only, no NAX hardware requirement
    assert m4_ksplit_eligible(4, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(5, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(4, 5120, 17408, 8, 64, dt)
    # m16: K % 256, N % 32, 4-bit, M in 1..16 (and NAX hardware)
    expect = nax_available()
    assert m16_nax_eligible(5, 5120, 17408, 4, 64, dt) == expect
    assert m16_nax_eligible(16, 17408, 5120, 4, 64, dt) == expect
    assert not m16_nax_eligible(17, 5120, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120 + 64, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120, 17408 + 8, 4, 64, dt)


def _quantized_fixture(K: int, N: int):
    mx.random.seed(3)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _stock(x, w_q, scales, biases):
    return mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
    )


def test_m4_kernel_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((4, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m4(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (4, N)
    assert diff < 0.25, f"m4 kernel drift too large: {diff}"


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_m16_nax_kernel_pads_and_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 16):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"nax16 kernel drift too large at M={m}: {diff}"


def test_qlinear_patch_routes_only_verify_shapes() -> None:
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        for m in (1, 3, 4, 8, 17, 64):
            x = (mx.random.normal((m, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
            y = layer(x)
            mx.eval(y)
            assert y.shape == (m, 256)
    finally:
        uninstall_nax_qlinear_patch()


def test_turbo_profile_carries_nax_env() -> None:
    from mtplx.profiles import PROFILES, PROFILE_CHOICES, apply_profile_env, restore_profile_env
    import os

    assert "turbo" in PROFILE_CHOICES
    profile = PROFILES["turbo"]
    assert profile.env_dict().get("MTPLX_NAX_VERIFY") == "1"
    assert profile.product_claim_eligible is False
    # Sustained env must be a subset (turbo = sustained + kernels).
    sustained = PROFILES["sustained"].env_dict()
    turbo = profile.env_dict()
    missing = {k: v for k, v in sustained.items() if turbo.get(k) != v}
    assert not missing, f"turbo drops sustained env keys: {missing}"
    previous = apply_profile_env("turbo")
    try:
        assert os.environ.get("MTPLX_NAX_VERIFY") == "1"
    finally:
        restore_profile_env(previous)
        assert os.environ.get("MTPLX_NAX_VERIFY") != "1"


def test_qlinear_patch_never_routes_in_prefill_phase() -> None:
    """Regression guard: prefill must stay on stock kernels byte-for-byte."""
    import mlx.core as mx
    from mtplx.attention_context import attention_phase
    from mtplx import nax_verify

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0, "m16": 0}
    orig_m4, orig_m16 = nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16

    def count_m4(*a, **k):
        calls["m4"] += 1
        return orig_m4(*a, **k)

    def count_m16(*a, **k):
        calls["m16"] += 1
        return orig_m16(*a, **k)

    nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = count_m4, count_m16
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("prefill"):
            mx.eval(layer(x))
        assert calls == {"m4": 0, "m16": 0}, f"kernels routed during prefill: {calls}"
        with attention_phase("decode_verify"):
            mx.eval(layer(x))
        assert calls["m4"] == 1, f"m4 kernel did not engage outside prefill: {calls}"
    finally:
        nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = orig_m4, orig_m16
        uninstall_nax_qlinear_patch()


def test_m6_kernel_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import m6_ksplit_eligible, nax_qmm_m6

    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 6):
        assert m6_ksplit_eligible(m, K, N, 4, 64, mx.bfloat16)
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m6(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"m6 kernel drift too large at M={m}: {diff}"
        if m == 5:
            exact = nax_qmm_m6(
                x,
                w_q,
                scales,
                biases,
                group_size=64,
                exact_m5=True,
            )
            mx.eval(exact)
            assert exact.shape == (m, N)
            assert mx.array_equal(exact, y).item()
        else:
            kp1 = nax_qmm_m6(
                x, w_q, scales, biases, group_size=64, k_parts=1
            )
            direct = nax_qmm_m6(
                x,
                w_q,
                scales,
                biases,
                group_size=64,
                k_parts=1,
                barrier_free_kp1=True,
            )
            mx.eval(kp1, direct)
            assert mx.array_equal(kp1, direct).item()
    assert not m6_ksplit_eligible(4, K, N, 4, 64, mx.bfloat16)
    assert not m6_ksplit_eligible(7, K, N, 4, 64, mx.bfloat16)


def test_m6_kp1_can_select_barrier_free_kernel(monkeypatch) -> None:
    from mtplx import nax_verify

    observed = {}

    class FakeKernel:
        def __call__(self, *, output_shapes, output_dtypes, **_kwargs):
            return [mx.zeros(output_shapes[0], dtype=output_dtypes[0])]

    def build(*_args, **kwargs):
        observed.update(kwargs)
        return FakeKernel()

    monkeypatch.setattr(nax_verify, "_build_kernel_m6_ksplit_np", build)
    y = nax_verify.nax_qmm_m6(
        mx.zeros((6, 32), dtype=mx.bfloat16),
        mx.zeros((64, 4), dtype=mx.uint32),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        group_size=32,
        k_parts=1,
        barrier_free_kp1=True,
    )
    assert y.shape == (6, 64)
    assert observed["barrier_free_kp1"] is True


def test_m5_m6_kernel_can_specialize_compile_time_k(monkeypatch) -> None:
    from mtplx import nax_verify

    observed = {}

    class FakeKernel:
        def __call__(self, *, template, output_shapes, output_dtypes, **_kwargs):
            observed["template"] = template
            return [mx.zeros(output_shapes[0], dtype=output_dtypes[0])]

    def build(*_args, **kwargs):
        observed.update(kwargs)
        return FakeKernel()

    monkeypatch.setattr(nax_verify, "_build_kernel_m6_ksplit_np", build)
    nax_verify.nax_qmm_m6(
        mx.zeros((5, 32), dtype=mx.bfloat16),
        mx.zeros((64, 4), dtype=mx.uint32),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        mx.zeros((64, 1), dtype=mx.bfloat16),
        group_size=32,
        exact_m5=True,
        compile_time_k=True,
    )
    assert observed["kconst"] == 32
    assert observed["template"] == [("T", mx.bfloat16), ("KCONST", 32)]


def test_qwen38_barrier_free_m6_kp1_configuration() -> None:
    from mtplx.nax_verify import configure_qwen38_m6_barrier_free_kp1

    assert configure_qwen38_m6_barrier_free_kp1(active=True) == {"active": True}
    assert configure_qwen38_m6_barrier_free_kp1(active=False) == {"active": False}


def test_qwen38_m5_m6_kconst_configuration() -> None:
    from mtplx.nax_verify import configure_qwen38_m56_kconst

    assert configure_qwen38_m56_kconst(active=True) == {
        "active": True,
        "m5_shapes": [[5120, 48], [5120, 10240]],
        "m6_shapes": [[5120, 10240]],
    }
    assert configure_qwen38_m56_kconst(active=False) == {
        "active": False,
        "m5_shapes": [],
        "m6_shapes": [],
    }


def test_m8_output_can_be_removed_without_removing_m7_or_mlp() -> None:
    from mtplx.nax_verify import configure_qwen38_m8_nax_island

    report = configure_qwen38_m8_nax_island(
        active=True,
        include_m8_output=False,
        include_m7_output=True,
        include_m8_mlp=True,
    )
    try:
        assert report["include_m8_output"] is False
        assert [6144, 5120] not in report["shapes"]
        assert report["m7_shapes"] == [[6144, 5120]]
        assert [5120, 17408] in report["shapes"]
    finally:
        configure_qwen38_m8_nax_island(active=False)


def test_qwen38_shape_specific_nax_split_tuning() -> None:
    from mtplx.nax_verify import configure_qwen38_nax_split_tuning

    report = configure_qwen38_nax_split_tuning(
        active=True,
        m7_linear_z_nsg4=True,
        m8_kv_nsg16=True,
        m8_qkv_nsg4=True,
    )
    try:
        assert report == {
            "active": True,
            "m7_nsg_by_shape": {"5120x6144": 4},
            "m8_nsg_by_shape": {"5120x1024": 16, "5120x10240": 4},
        }
    finally:
        configure_qwen38_nax_split_tuning(active=False)


def test_qwen38_shape_specific_m5_m6_partition_candidate() -> None:
    from mtplx.nax_verify import configure_qwen38_m56_partition_tuning

    report = configure_qwen38_m56_partition_tuning(active=True)
    try:
        assert report["m5_kparts_by_shape"] == {
            "5120x48": 4,
            "5120x1024": 1,
            "5120x17408": 1,
            "17408x5120": 1,
        }
        assert report["m6_kparts_by_shape"] == {
            "5120x1024": 1,
            "5120x10240": 1,
            "5120x12288": 4,
            "5120x17408": 2,
            "17408x5120": 4,
        }
    finally:
        configure_qwen38_m56_partition_tuning(active=False)

    m6_only = configure_qwen38_m56_partition_tuning(active=True, m5_active=False)
    try:
        assert m6_only["m5_kparts_by_shape"] == {}
        assert m6_only["m6_kparts_by_shape"]
    finally:
        configure_qwen38_m56_partition_tuning(active=False)


def test_vk_6bit_hexpack_ksplit_matches_stock() -> None:
    """The 9B-tier 6-bit lane (2026-07-07): MLX packs 6-bit values
    bit-contiguously little-endian; the hexpack kernels must agree with
    stock quantized_matmul within the accumulation-order ULP band."""
    from mtplx.verify_kernels import (
        vk_eligible_ksplit,
        vk_qmm_m4_ksplit,
        vk_qmm_m6_ksplit,
    )

    K, N = 4096, 1024
    for dtype in (mx.bfloat16, mx.float16):
        for gs in (32, 64, 128):
            mx.random.seed(5)
            w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
            w_q, scales, biases = mx.quantize(w, group_size=gs, bits=6)
            mx.eval(w_q, scales, biases)
            for m, fn in ((4, vk_qmm_m4_ksplit), (5, vk_qmm_m6_ksplit), (6, vk_qmm_m6_ksplit)):
                assert vk_eligible_ksplit(m, K, N, 6, gs, dtype)
                x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(dtype)
                y = fn(x, w_q, scales, biases, bits=6, group_size=gs)
                ref = mx.quantized_matmul(
                    x, w_q, scales=scales, biases=biases,
                    transpose=True, group_size=gs, bits=6,
                )
                diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
                assert y.shape == (m, N)
                assert diff < 0.05, f"6-bit drift {dtype} gs={gs} M={m}: {diff}"


def test_qlinear_patch_routes_6bit_verify_shapes() -> None:
    """The patch routes 6-bit verify shapes (N >= 2048 floor) through the
    hexpack kernels and leaves small-N projections on stock."""
    from mtplx import verify_kernels

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0}
    orig = verify_kernels.vk_qmm_m4_ksplit

    def counting(*a, **k):
        calls["m4"] += 1
        return orig(*a, **k)

    from mtplx.attention_context import attention_phase

    import mtplx.nax_verify  # noqa: F401  (patch reads through the module)

    verify_kernels.vk_qmm_m4_ksplit = counting
    try:
        big = nn.QuantizedLinear(512, 2048, bias=False, group_size=64, bits=6)
        small = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=6)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "6-bit verify shape did not route the hexpack kernel"
            mx.eval(small(x))
            assert calls["m4"] == 1, "small-N 6-bit projection must stay stock"
        with attention_phase("prefill"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "prefill must stay stock"
    finally:
        verify_kernels.vk_qmm_m4_ksplit = orig
        uninstall_nax_qlinear_patch()
