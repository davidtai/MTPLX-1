"""MTPLX_FABLE_HC_M4 — wiring, eligibility, and the eager chain's op contract.

Two things are proved here, both entirely on the CPU stream with tiny tensors
(no Metal, no kernel dispatch, no model):

1. THE OP CONTRACT the Metal kernel encodes. ``mtplx/kernels/qwen4_m4_hyper_read``
   reproduces the eager chain op by op, and every rounding boundary it copies
   is an assumption about how MLX types and rounds a specific expression. If
   MLX ever changes one of those (``2.0 * bf16`` promoting to fp32, ``mx.mean``
   accumulating in fp32, ``nn.silu`` growing its own kernel), the kernel goes
   silently wrong. These tests fail instead.

2. THE GATE. Flag off: nothing changes and nothing is imported. Flag on: rows
   1 keeps the old path, rows 2..8 either take the kernel or RAISE with the
   offending field named. There is no silent fallback — that is the failure
   mode that left MTPLX_FUSED_HC_V3 armed but structurally inert at M=4.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.kernels.qwen4_m4_hyper_read as hcm4
import mtplx.models.qwen4_exp as qwen4_exp
import mtplx.runtime_options as runtime_options

HCD = hcm4.HCD
LOWRANK = hcm4.R_LOWRANK
HC = hcm4.HC


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    """Default every test to the shipped state, whatever the session env is."""

    monkeypatch.setattr(runtime_options, "_FABLE_HC_M4", False)


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(runtime_options, "_FABLE_HC_M4", True)
    return True


def _bits(value: mx.array) -> mx.array:
    widths = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}
    return value.view(widths[value.dtype])


def _same_bits(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(_bits(a) == _bits(b)).item())


# --------------------------------------------------------------------------
# 1. the eager chain's op contract, as encoded in the Metal source
# --------------------------------------------------------------------------


def test_divide_by_hc_count_stays_bf16():
    """``mix / self.hc_count`` must not widen: the kernel rounds it to bf16."""

    a = (mx.random.normal((64,)) * 3).astype(mx.bfloat16)
    assert (a / 4).dtype == mx.bfloat16
    assert _same_bits(a / 4, (a.astype(mx.float32) * 0.25).astype(mx.bfloat16))


def test_inject_scale_stays_bf16():
    """``2.0 * mx.sigmoid(...)`` — a Python float is a weak scalar in MLX, so
    the inject value the kernel writes is bf16, not fp32."""

    a = (mx.random.normal((64,)) * 3).astype(mx.bfloat16)
    s = mx.sigmoid(a)
    assert (2.0 * s).dtype == mx.bfloat16


def test_silu_is_x_times_sigmoid_x():
    """The kernel writes ``(T)(sigmoid(t0) * t0)``; nn.silu must be that."""

    a = (mx.random.normal((256,)) * 3).astype(mx.bfloat16)
    assert nn.silu(a).dtype == mx.bfloat16
    assert _same_bits(nn.silu(a), a * mx.sigmoid(a))


def test_hc_mean_is_bf16_sum_times_quarter():
    """``mx.mean(mix * grouped, axis=-2)`` == ``mx.sum(...) * 0.25``, and that
    sum accumulates IN bf16 — which is why the kernel's hc reduction rounds at
    every add instead of accumulating in fp32."""

    a = (mx.random.normal((512, HC, 4)) * 3).astype(mx.bfloat16)
    m = mx.mean(a, axis=-2)
    s = mx.sum(a, axis=-2)
    assert m.dtype == mx.bfloat16 and s.dtype == mx.bfloat16
    assert _same_bits(m, (s.astype(mx.float32) * 0.25).astype(mx.bfloat16))

    seq = a[:, 0, :]
    for g in range(1, HC):
        seq = seq + a[:, g, :]
    assert _same_bits(s, seq), (
        "mx.sum over a length-4 bf16 axis is no longer a sequential bf16 "
        "accumulation; qwen4_m4_hyper_read's hc reduction must follow it"
    )


def test_hc_mean_is_not_fp32_accumulated():
    """Guard the *reason* the test above is not cosmetic: an fp32 accumulation
    of the same four terms is a genuinely different answer."""

    a = (mx.random.normal((4096, HC)) * 3).astype(mx.bfloat16)
    s = mx.sum(a, axis=-1)
    f32 = a.astype(mx.float32).sum(axis=-1).astype(mx.bfloat16)
    assert not _same_bits(s, f32)


def test_grouped_rms_norm_decomposition():
    """``GroupedRMSNorm`` == per-group rms_norm rounded to bf16, then a bf16
    multiply by the full-width weight — the two-step the kernel's K0 copies."""

    dims, group = 32, 8
    norm = qwen4_exp.GroupedRMSNorm(dims, group, eps=1e-6)
    norm.weight = (mx.random.normal((dims,)) * 0.5 + 1.0).astype(mx.bfloat16)
    x = (mx.random.normal((3, dims)) * 2).astype(mx.bfloat16)

    got = norm(x)
    grouped = x.reshape(3, -1, group)
    step1 = mx.fast.rms_norm(grouped, None, 1e-6).reshape(3, dims)
    assert step1.dtype == mx.bfloat16
    assert _same_bits(got, step1 * norm.weight)


# --------------------------------------------------------------------------
# 2. check_shapes — returns rows, or raises with the field named
# --------------------------------------------------------------------------


def _family_weights(dtype=mx.bfloat16, *, down_rows=LOWRANK):
    return {
        "gamma": mx.zeros((HCD,), dtype=dtype),
        "wd": mx.zeros((down_rows, HCD), dtype=dtype),
        "wu": mx.zeros((HCD, LOWRANK), dtype=dtype),
        "wi": mx.zeros((HC, HCD), dtype=dtype),
    }


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8])
def test_check_shapes_accepts_verify_widths(rows):
    w = _family_weights()
    x = mx.zeros((rows, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == rows


@pytest.mark.parametrize("rows", [1, 9, 16])
def test_check_shapes_rejects_other_widths(rows):
    w = _family_weights()
    x = mx.zeros((rows, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="wired for 2..8 rows"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_rejects_wrong_hidden():
    w = _family_weights()
    with pytest.raises(ValueError, match=r"must be \[\.\.\., 10240\]"):
        hcm4.check_shapes(
            mx.zeros((4, 4096), dtype=mx.bfloat16),
            w["gamma"], w["wd"], w["wu"], w["wi"],
        )


def test_check_shapes_names_the_bad_weight():
    w = _family_weights(down_rows=256)
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_down.weight"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_rejects_dtype_mismatch():
    w = _family_weights()
    w["wu"] = mx.zeros((HCD, LOWRANK), dtype=mx.float16)
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_up.weight dtype"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_takes_the_module_shape_unreshaped():
    """GatedResidual is called with [B, S, 10240]; R is the leading product,
    and validating must not require materialising a reshape node."""

    w = _family_weights()
    x = mx.zeros((1, 4, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == 4
    x = mx.zeros((2, 3, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == 6


def test_driver_requires_a_two_d_view():
    w = _family_weights()
    with pytest.raises(ValueError, match="wants a 2-D"):
        hcm4.fused_hc_read_m4(
            mx.zeros((1, 4, HCD), dtype=mx.bfloat16),
            w["gamma"], w["wd"], w["wu"], w["wi"],
        )


def test_check_shapes_allows_missing_inject():
    """The trunk mixer is built with ``use_combine=False``."""

    w = _family_weights()
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], None) == 4


def test_weight_bytes_matches_the_census_figure():
    assert hcm4.weight_bytes_per_read(2) == 13_209_600
    assert hcm4.weight_bytes_per_read(2, has_inject=False) == 13_127_680


# --------------------------------------------------------------------------
# 3. the gate in GatedResidual
# --------------------------------------------------------------------------


def _family_module(use_combine=True, dtype=mx.bfloat16):
    mod = qwen4_exp.GatedResidual(qwen4_exp.TextArgs(), use_combine=use_combine)
    w = _family_weights(dtype)
    mod.hc_norm.weight = w["gamma"]
    mod.input_mix_weight_down.weight = w["wd"]
    mod.input_mix_weight_up.weight = w["wu"]
    if use_combine:
        mod.block_inject_weight.weight = w["wi"]
    return mod


def _small_module(use_combine=True):
    args = qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4, rms_norm_eps=1e-6)
    mod = qwen4_exp.GatedResidual(args, use_combine=use_combine)
    hcd = args.hc_count * args.hidden_size
    mod.hc_norm.weight = (mx.random.normal((hcd,)) * 0.3 + 1.0).astype(mx.bfloat16)
    mod.input_mix_weight_down.weight = (
        mx.random.normal((args.hc_lowrank, hcd)) * 0.1
    ).astype(mx.bfloat16)
    mod.input_mix_weight_up.weight = (
        mx.random.normal((hcd, args.hc_lowrank)) * 0.1
    ).astype(mx.bfloat16)
    if use_combine:
        mod.block_inject_weight.weight = (
            mx.random.normal((args.hc_count, hcd)) * 0.1
        ).astype(mx.bfloat16)
    return mod, args


def test_flag_off_never_applies():
    mod = _family_module()
    for rows in (1, 2, 4, 8, 32):
        x = mx.zeros((1, rows, HCD), dtype=mx.bfloat16)
        assert mod._hc_m4_applies(x) is False


def test_flag_off_leaves_the_eager_chain_bit_identical():
    """The full eager expression, transcribed, on a tiny config."""

    mod, args = _small_module()
    hcd = args.hc_count * args.hidden_size
    x = (mx.random.normal((1, 4, hcd)) * 2).astype(mx.bfloat16)

    mixed, passthrough, inject = mod(x)

    normed = mod.hc_norm(x)
    mix = nn.silu(mod.input_mix_weight_down(normed) / args.hc_count)
    mix = mx.sigmoid(mod.input_mix_weight_up(mix))
    mix = mix.reshape(*mix.shape[:-1], args.hc_count, args.hidden_size)
    grouped = normed.reshape(*normed.shape[:-1], args.hc_count, args.hidden_size)
    want_mixed = mx.mean(mix * grouped, axis=-2)
    want_inject = 2.0 * mx.sigmoid(mod.block_inject_weight(normed) / args.hc_count)

    assert _same_bits(mixed, want_mixed)
    assert _same_bits(inject, want_inject)
    assert passthrough is x


def test_flag_off_tolerates_off_family_geometry():
    """Construction must not raise for a non-Flash-Next config when the flag
    is unset — this module class is shared."""

    qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4))


@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_armed_applies_at_verify_widths(armed, rows):
    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, rows, HCD), dtype=mx.bfloat16)) is True


def test_armed_applies_to_the_noncombine_mixer(armed):
    mod = _family_module(use_combine=False)
    assert mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16)) is True


@pytest.mark.parametrize("rows", [1])
def test_armed_leaves_row_one_alone(armed, rows):
    """rows == 1 is the draft path's business (v3 / eager); this gate must not
    take it, and must not raise on it either."""

    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, rows, HCD), dtype=mx.bfloat16)) is False


def test_armed_leaves_prefill_widths_alone(armed):
    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, 64, HCD), dtype=mx.bfloat16)) is False


def test_armed_off_family_config_raises_at_construction(armed):
    with pytest.raises(RuntimeError, match="not the Flash-Next family shape"):
        qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4))
    with pytest.raises(RuntimeError, match="hc_count=2"):
        qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hc_count=2))


def test_armed_quantized_mix_weights_raise(armed):
    mod = _family_module()
    mod.input_mix_weight_down.scales = mx.zeros((LOWRANK, HCD // 64), mx.bfloat16)
    with pytest.raises(RuntimeError, match="input_mix_weight_down is quantized"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_armed_dtype_mismatch_raises(armed):
    """fp32 mix weights against a bf16 hyper state: raise, do not fall back."""

    mod = _family_module(dtype=mx.float32)
    with pytest.raises(ValueError, match="dtype"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_armed_wrong_down_rows_raise(armed):
    mod = _family_module()
    mod.input_mix_weight_down.weight = mx.zeros((256, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_down.weight"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_env_flag_is_read_once(monkeypatch):
    """A mid-run env change must not reach the hot path: two traces of the
    same compiled verify graph would then disagree about which read they
    contain."""

    before = runtime_options.fable_hc_m4_enabled()
    monkeypatch.setenv("MTPLX_FABLE_HC_M4", "1")
    assert runtime_options.fable_hc_m4_enabled() is before
    monkeypatch.delenv("MTPLX_FABLE_HC_M4", raising=False)
    assert runtime_options.fable_hc_m4_enabled() is before


def test_env_flag_defaults_off():
    from mtplx.runtime_options import env_bool

    assert env_bool("MTPLX_FABLE_HC_M4", default=False, env={}) is False


def test_env_flag_rejects_a_bad_spelling():
    from mtplx.runtime_options import env_bool

    with pytest.raises(ValueError):
        env_bool("MTPLX_FABLE_HC_M4", default=False, env={"MTPLX_FABLE_HC_M4": "yep"})


# --------------------------------------------------------------------------
# 4. the driver's own guards (no dispatch: these all raise before the kernel)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"norm_threads": 100}, "norm_threads"),
        ({"down_threads": 2048}, "down_threads"),
        ({"out_per_tg": 0}, "out_per_tg"),
        ({"d_per_block": 999}, "d_per_block"),
    ],
)
def test_driver_rejects_bad_tuning(kwargs, match):
    w = _family_weights()
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match=match):
        hcm4.fused_hc_read_m4(
            x, w["gamma"], w["wd"], w["wu"], w["wi"], **kwargs
        )


def test_dispatch_budget_is_three():
    assert hcm4.DISPATCHES_PER_READ == 3
    assert hcm4.EAGER_DISPATCHES_PER_READ == 11
