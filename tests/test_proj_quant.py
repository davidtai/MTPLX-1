"""Load-time resident quantization: scope, config validation, flag plumbing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    parse_memory_bytes,
    proj_quant_plan_discount,
)
from mtplx.expert_streaming_models import (
    get_model_spec,
    plan_expert_memory,
    proj_quant_covers,
    affine_quant_kept_bytes,
)
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.resident_loader import (
    ResidentLoadError,
    _runtime_quantize_projections,
    _runtime_requantize_projections,
)


def _tiny_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("bench_depth_matrix_rq", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_depth_matrix_rq", module)
    spec.loader.exec_module(module)
    return module


def _config_kwargs(**overrides):
    kwargs = {
        "model_key": "hy3-expert-q2",
        "memory_limit_bytes": 96 * 1024**3,
        "max_live_kv_tokens": 4096,
    }
    kwargs.update(overrides)
    return kwargs


def test_proj_quant_config_accepts_supported_modes() -> None:
    for mode in (None, "q8", "q4"):
        config = ExpertStreamingConfig(**_config_kwargs(proj_quant=mode))
        assert config.proj_quant == mode


def test_proj_quant_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="proj_quant"):
        ExpertStreamingConfig(**_config_kwargs(proj_quant="int8"))


def test_runtime_quantize_scopes_attention_shared_and_dense_mlp() -> None:
    args = _tiny_args()
    model = Hy3Model(args)
    original_q_weight = model.model.layers[1].self_attn.q_proj.weight.astype(
        mx.float32
    )

    quantized = _runtime_quantize_projections(model, "q4")

    expected = set()
    for layer in range(args.num_hidden_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected.add(f"model.layers.{layer}.self_attn.{proj}")
    # Layer 0 is dense (first_k_dense_replace=1); layer 1 carries the
    # shared expert.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expected.add(f"model.layers.0.mlp.{proj}")
        expected.add(f"model.layers.1.mlp.shared_mlp.{proj}")
    assert set(quantized) == expected

    for path in expected:
        module = model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        assert isinstance(module, nn.QuantizedLinear), path
        assert module.bits == 4 and module.group_size == 64

    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)
    assert isinstance(model.model.embed_tokens, nn.Embedding)

    # Dequantized projection stays close to the BF16 original.
    x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
    reference = mx.matmul(x.astype(mx.float32), original_q_weight.T)
    actual = model.model.layers[1].self_attn.q_proj(x).astype(mx.float32)
    mx.eval(reference, actual)
    error = float(
        (mx.abs(actual - reference).mean() / (mx.abs(reference).mean() + 1e-6)).item()
    )
    assert error < 0.2, f"q4 dequantization drifted: {error}"


def test_runtime_quantize_rejects_matchless_model() -> None:
    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(8, 8, bias=False)

    with pytest.raises(ResidentLoadError, match=r"matched no trunk \*_proj"):
        _runtime_quantize_projections(Bare(), "q4")


@pytest.mark.parametrize(
    ("path", "covered"),
    [
        ("model.layers.3.self_attn.q_proj", True),
        ("model.layers.3.self_attn.o_proj", True),
        ("model.layers.1.mlp.shared_mlp.gate_proj", True),
        ("model.layers.1.mlp.shared_mlp.gate_up_proj", True),
        ("model.layers.0.mlp.down_proj", True),
        ("model.layers.1.mlp.router.gate", False),
        ("model.layers.1.mlp.gate", False),
        ("lm_head", False),
        ("model.embed_tokens", False),
        ("model.layers.1.input_layernorm", False),
        ("model.layers.80.eh_proj", False),
    ],
)
def test_proj_quant_scope(path: str, covered: bool) -> None:
    assert proj_quant_covers(path) is covered


def test_affine_quant_kept_bytes_matches_group64_affine() -> None:
    # 1 MiB of BF16 = 512 Ki elements; q4 packs to 256 KiB + 32 KiB of
    # BF16 scales and biases (one pair per 64-element group).
    assert affine_quant_kept_bytes(1024 * 1024, "q4") == 256 * 1024 + 32 * 1024
    assert affine_quant_kept_bytes(1024 * 1024, "q8") == 512 * 1024 + 32 * 1024


def test_plan_discount_shrinks_fixed_bytes_exactly() -> None:
    spec = get_model_spec("hy3-expert-q2")
    base = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=4096
    )
    discounted = plan_expert_memory(
        spec,
        total_limit_bytes=110 * 1024**3,
        context_tokens=4096,
        resident_discount_bytes=10 * 1024**3,
    )
    assert base.fixed_bytes - discounted.fixed_bytes == 10 * 1024**3
    with pytest.raises(ValueError, match="exceeds the resident footprint"):
        plan_expert_memory(
            spec,
            total_limit_bytes=110 * 1024**3,
            context_tokens=4096,
            resident_discount_bytes=spec.resident_bytes,
        )


def test_manifest_plan_discount_scopes_and_prices_tensors() -> None:
    from types import SimpleNamespace

    tensors = (
        SimpleNamespace(
            tensor="model.layers.1.self_attn.q_proj.weight",
            dtype="BF16",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # router stays exact
            tensor="model.layers.1.mlp.router.gate.weight",
            dtype="BF16",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # non-BF16 never discounts
            tensor="model.layers.2.self_attn.q_proj.weight",
            dtype="F32",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # biases keep loaded precision
            tensor="model.layers.1.self_attn.q_proj.bias",
            dtype="BF16",
            length=4096,
        ),
    )
    manifest = SimpleNamespace(resident_tensors=tensors)
    assert proj_quant_plan_discount(manifest, None) == 0
    expected = 1024 * 1024 - affine_quant_kept_bytes(1024 * 1024, "q4")
    assert proj_quant_plan_discount(manifest, "q4") == expected


def test_kv_quant_config_validation() -> None:
    for mode in (None, "q8", "q4"):
        assert ExpertStreamingConfig(**_config_kwargs(kv_quant=mode)).kv_quant == mode
    with pytest.raises(ValueError, match="kv_quant"):
        ExpertStreamingConfig(**_config_kwargs(kv_quant="int8"))


def test_plan_prices_quantized_kv() -> None:
    spec = get_model_spec("hy3-expert-q2")
    base = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=8192
    )
    quant = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=8192, kv_quant="q4"
    )
    raw = 8192 * spec.kv_bytes_per_token
    assert base.kv_bytes == raw
    assert quant.kv_bytes == affine_quant_kept_bytes(raw, "q4")
    with pytest.raises(ValueError, match="kv_quant"):
        plan_expert_memory(
            spec, total_limit_bytes=110 * 1024**3, context_tokens=8192,
            kv_quant="int8",
        )


def test_plan_prices_prefetch_ring_on_the_fixed_side() -> None:
    spec = get_model_spec("hy3-expert-q2")
    base = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=4096,
        island_layer_count=49,
    )
    ring = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=4096,
        island_layer_count=49, prefetch_slots_per_layer=8,
    )
    streamed = spec.routed_layer_count - 49
    expected = streamed * 8 * spec.expert_record_bytes
    assert ring.prefetch_bytes == expected
    assert ring.fixed_bytes - base.fixed_bytes == expected
    assert ring.prefetch_slots_per_layer == 8


def test_kv_quant_guard_rejects_models_that_ignore_it() -> None:
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    from mtplx.resident_loader import _verify_kv_quant_honored

    class _IgnoringModel(nn.Module):
        def make_cache(self):
            return [KVCache()]

    class _HonoringModel(nn.Module):
        def make_cache(self):
            return [QuantizedKVCache(group_size=64, bits=4)]

    with pytest.raises(ResidentLoadError, match="ignores it"):
        _verify_kv_quant_honored(_IgnoringModel(), "q4")
    _verify_kv_quant_honored(_HonoringModel(), "q4")

    class _BrokenModel(nn.Module):
        def make_cache(self):
            raise RuntimeError("boom")

    with pytest.raises(ResidentLoadError, match="probe failed"):
        _verify_kv_quant_honored(_BrokenModel(), "q4")


def test_make_cache_honors_kv_quant_attribute() -> None:
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    model = Hy3Model(_tiny_args())
    default = model.make_cache()
    assert len(default) == 2
    assert all(type(c) is KVCache for c in default)

    model._mtplx_kv_quant = "q4"
    quantized = model.make_cache()
    assert all(type(c) is QuantizedKVCache for c in quantized)
    assert all(c.bits == 4 and c.group_size == 64 for c in quantized)


def test_hy3_attention_matches_between_stock_and_quantized_cache() -> None:
    """Decode-shaped forward parity: q8 KV should track the stock cache
    closely (loose tolerance — quantization noise, not correctness)."""

    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    args = _tiny_args()
    # mx.quantize needs group_size >= 32 and group_size | head_dim.
    args.head_dim = 32
    model = Hy3Model(args)

    class _NullMoE(nn.Module):
        """The sparse layer's experts need a bound runtime; attention
        parity does not."""

        def __call__(self, x: mx.array) -> mx.array:
            return x * 0

    model.model.layers[1].mlp = _NullMoE()
    tokens = mx.array([[3, 5, 7, 11, 2, 9, 4, 8]], dtype=mx.int32)
    step = mx.array([[6]], dtype=mx.int32)

    stock = [KVCache() for _ in model.layers]
    quant = [QuantizedKVCache(group_size=32, bits=8) for _ in model.layers]
    mx.eval(model(tokens, cache=stock))
    mx.eval(model(tokens, cache=quant))
    a = model(step, cache=stock).astype(mx.float32)
    b = model(step, cache=quant).astype(mx.float32)
    mx.eval(a, b)
    scale = float(mx.abs(a).mean().item()) + 1e-6
    drift = float(mx.abs(a - b).mean().item()) / scale
    assert drift < 0.05, f"quantized-KV forward drifted: {drift}"


def test_hy3_attention_matches_between_stock_and_quantized_cache_q4() -> None:
    """Same decode-shaped forward parity as the q8 test above, at q4 --
    the coarser mode the pricing (kv_bytes_per_token_for) and the
    presets.toml championship lane both need to actually be usable, not
    just cheap on paper."""

    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    args = _tiny_args()
    args.head_dim = 32
    model = Hy3Model(args)

    class _NullMoE(nn.Module):
        def __call__(self, x: mx.array) -> mx.array:
            return x * 0

    model.model.layers[1].mlp = _NullMoE()
    tokens = mx.array([[3, 5, 7, 11, 2, 9, 4, 8]], dtype=mx.int32)
    step = mx.array([[6]], dtype=mx.int32)

    stock = [KVCache() for _ in model.layers]
    quant = [QuantizedKVCache(group_size=32, bits=4) for _ in model.layers]
    mx.eval(model(tokens, cache=stock))
    mx.eval(model(tokens, cache=quant))
    a = model(step, cache=stock).astype(mx.float32)
    b = model(step, cache=quant).astype(mx.float32)
    mx.eval(a, b)
    scale = float(mx.abs(a).mean().item()) + 1e-6
    drift = float(mx.abs(a - b).mean().item()) / scale
    # q4 has 4x fewer codes than q8 (test above): a looser but still tight
    # bound. Empirically observed drift for this fixture is well under 0.3;
    # 1.0 (bounded by _NullMoE's exact-zero second layer) would mean the
    # cache silently returned garbage, not merely coarser codes.
    assert drift < 0.3, f"quantized-KV (q4) forward drifted: {drift}"


@pytest.mark.parametrize("kv_quant", (None, "q8", "q4"))
def test_quantized_kv_cache_round_trip_write_read_parity(kv_quant: str | None) -> None:
    """Direct write/read parity through the SAME cache classes
    Hy3Model.make_cache constructs (mlx_lm KVCache / QuantizedKVCache,
    group_size=64), shaped like the real Hy3 GQA config (num_key_value_heads=8,
    head_dim=128, bf16 activations): 'off' must be bit-exact; q8/q4 must
    recover the source K/V within their quantization step, not merely
    "close enough to not crash"."""

    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    mx.random.seed(1234)
    batch, n_kv_heads, seq, head_dim = 1, 8, 6, 128
    keys = mx.random.normal((batch, n_kv_heads, seq, head_dim)).astype(mx.bfloat16)
    values = mx.random.normal((batch, n_kv_heads, seq, head_dim)).astype(mx.bfloat16)
    mx.eval(keys, values)

    if kv_quant is None:
        cache = KVCache()
        out_keys, out_values = cache.update_and_fetch(keys, values)
        assert mx.array_equal(out_keys, keys).item()
        assert mx.array_equal(out_values, values).item()
        return

    bits = {"q8": 8, "q4": 4}[kv_quant]
    # Empirically measured max abs error for this exact shape/seed/dtype:
    # q8 ~0.031, q4 ~0.344 (group_size=64, bf16 scale+bias). Thresholds
    # below carry real margin while still catching a broken round trip.
    max_abs_error = {"q8": 0.2, "q4": 0.75}[kv_quant]
    cache = QuantizedKVCache(group_size=64, bits=bits)
    (packed_k, scale_k, bias_k), (packed_v, scale_v, bias_v) = (
        cache.update_and_fetch(keys, values)
    )
    dequant_k = mx.dequantize(
        packed_k, scales=scale_k, biases=bias_k, group_size=64, bits=bits
    )
    dequant_v = mx.dequantize(
        packed_v, scales=scale_v, biases=bias_v, group_size=64, bits=bits
    )
    mx.eval(dequant_k, dequant_v)
    err_k = float(mx.abs(dequant_k.astype(mx.float32) - keys.astype(mx.float32)).max().item())
    err_v = float(mx.abs(dequant_v.astype(mx.float32) - values.astype(mx.float32)).max().item())
    assert err_k < max_abs_error, f"{kv_quant} K round-trip error too large: {err_k}"
    assert err_v < max_abs_error, f"{kv_quant} V round-trip error too large: {err_v}"
    # Not exact -- a quantized round trip must actually introduce some loss,
    # or this test would silently stop meaning anything if bits were ignored.
    assert err_k > 0.0 or err_v > 0.0


def test_mtp_draft_cache_stays_stock_kvcache_when_trunk_kv_quant_is_set() -> None:
    """Constraint check (acceptance law): kv_bits must apply to the trunk KV
    cache ONLY. The MTP NextN draft/verify head builds its own cache via a
    SEPARATE make_mtp_cache closure (mtplx/hy3_mtp_patch.py) that is
    hardcoded to stock KVCache and never reads _mtplx_kv_quant -- assert
    that stays true even when the trunk is actively running quantized KV."""

    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    from mtplx.hy3_mtp_patch import inject_hy3_streamed_mtp_support
    from mtplx.models.hy3_mlx import Hy3MTP

    args = _tiny_args()
    args.num_nextn_predict_layers = 1
    model = Hy3Model(args)
    model._mtplx_kv_quant = "q4"

    mtp = Hy3MTP(args, num_mtp_layers=1)
    injected = inject_hy3_streamed_mtp_support(
        model,
        Path("/unused-mtp-module-is-prebuilt"),
        {"model_type": "hy_v3"},
        mtp_module=mtp,
    )
    assert injected is True

    # The trunk DOES honor _mtplx_kv_quant (established by
    # test_make_cache_honors_kv_quant_attribute above) ...
    trunk_cache = model.make_cache()
    assert all(type(entry) is QuantizedKVCache for entry in trunk_cache)

    # ... but the MTP draft cache never does, regardless of the trunk's
    # kv_quant setting.
    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == len(mtp.layers) == 1
    assert all(type(entry) is KVCache for entry in mtp_cache)


def test_proj_quant_survives_benchmark_option_pipeline() -> None:
    """CLI vector -> parser -> runtime options -> config factory, mirroring
    the island-count guard (option keys have silently dropped between
    argparse and ExpertStreamingConfig before)."""

    from types import SimpleNamespace

    bench = _load_benchmark_module()
    parser = None
    import argparse as _argparse

    for name in dir(bench):
        fn = getattr(bench, name)
        if callable(fn) and name.endswith("parser"):
            candidate = fn()
            if isinstance(candidate, _argparse.ArgumentParser):
                parser = candidate
                break
    assert parser is not None, "benchmark parser factory not found"
    args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--proj-quant", "q4",
        ]
    )
    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(args),
    }
    assert options["proj_quant"] == "q4"
    apis = SimpleNamespace(
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
    )
    config = bench._runtime_config(apis, "hy3-expert-q2", options)
    assert config.proj_quant == "q4"

    kv_args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--kv-quant", "q4",
        ]
    )
    kv_options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(kv_args),
    }
    assert kv_options["kv_quant"] == "q4"
    kv_config = bench._runtime_config(apis, "hy3-expert-q2", kv_options)
    assert kv_config.kv_quant == "q4"

    with pytest.raises(SystemExit):
        bench.main(
            [
                "--model", "hy3-q2",
                "--hy3-q2-model-root", "/tmp",
                "--kv-quant", "q4",
                "--verify-strategy", "capture_commit",
            ]
        )

    for mode, per_read, at_open in (
        ("per-read", True, False),
        ("at-open", False, True),
        ("headers-only", False, False),
    ):
        integrity_args = parser.parse_args(
            [
                "--model", "hy3-q2",
                "--hy3-q2-model-root", "/tmp",
                "--memory-limit", "96GiB",
                "--expert-integrity", mode,
            ]
        )
        integrity_options = {
            **bench.DEFAULT_RUNTIME_OPTIONS,
            "trace_routes": False,
            **bench._runtime_options_from_args(integrity_args),
        }
        integrity_config = bench._runtime_config(
            apis, "hy3-expert-q2", integrity_options
        )
        assert integrity_config.verify_record_hashes is per_read, mode
        assert integrity_config.verify_sidecar_hash_at_open is at_open, mode


def test_prefetch_slots_survive_benchmark_option_pipeline() -> None:
    """Same CLI -> options -> config path as resident/kv-quant: the
    prefetch knob must not silently drop between argparse and
    ExpertStreamingConfig."""

    from types import SimpleNamespace

    bench = _load_benchmark_module()
    parser = None
    import argparse as _argparse

    for name in dir(bench):
        fn = getattr(bench, name)
        if callable(fn) and name.endswith("parser"):
            candidate = fn()
            if isinstance(candidate, _argparse.ArgumentParser):
                parser = candidate
                break
    assert parser is not None, "benchmark parser factory not found"
    assert bench.DEFAULT_RUNTIME_OPTIONS["prefetch_slots"] == 0

    apis = SimpleNamespace(
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
    )

    args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--prefetch-slots", "8",
        ]
    )
    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(args),
    }
    assert options["prefetch_slots"] == 8
    config = bench._runtime_config(apis, "hy3-expert-q2", options)
    assert config.prefetch_slots == 8

    default_args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
        ]
    )
    default_options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(default_args),
    }
    assert default_options["prefetch_slots"] == 0
    default_config = bench._runtime_config(apis, "hy3-expert-q2", default_options)
    assert default_config.prefetch_slots == 0


# --------------------------------------------------------------------------
# proj_requant: q8 -> q4 double quantization of PRE-quantized residents.
# --------------------------------------------------------------------------


def _prequantize_covered_to_q8(model) -> None:
    """Mimic an oq2e-style load: covered trunk *_proj Linears ship q8/gs64.

    Router gates, embeddings, lm_head, and norms stay BF16 (proj_quant_covers
    excludes them), so the model matches the checkpoint the requant experiment
    targets.
    """

    nn.quantize(
        model,
        group_size=64,
        bits=8,
        mode="affine",
        class_predicate=lambda path, module: (
            isinstance(module, nn.Linear)
            and not isinstance(module, nn.QuantizedLinear)
            and proj_quant_covers(path)
        ),
    )


def test_proj_requant_config_accepts_none_and_q4() -> None:
    for mode in (None, "q4"):
        config = ExpertStreamingConfig(**_config_kwargs(proj_requant=mode))
        assert config.proj_requant == mode


def test_proj_requant_config_rejects_q8_and_unknown() -> None:
    # q8 is a no-op (the sanctioned experiment is strictly q8 -> q4); reject it
    # alongside any other typo so nothing silently passes through.
    for bad in ("q8", "q2", "int8"):
        with pytest.raises(ValueError, match="proj_requant"):
            ExpertStreamingConfig(**_config_kwargs(proj_requant=bad))


def test_proj_requant_does_not_share_state_with_proj_quant() -> None:
    # Both may be set on one config without interaction; proj_requant must not
    # engage proj_quant's validation or vice versa.
    config = ExpertStreamingConfig(
        **_config_kwargs(proj_quant="q8", proj_requant="q4")
    )
    assert config.proj_quant == "q8"
    assert config.proj_requant == "q4"


def test_runtime_requantize_scopes_and_builds_standard_q4_modules() -> None:
    args = _tiny_args()
    model = Hy3Model(args)
    _prequantize_covered_to_q8(model)

    # The exact same scope proj_quant covers, now already-q8 QuantizedLinears.
    expected = set()
    for layer in range(args.num_hidden_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected.add(f"model.layers.{layer}.self_attn.{proj}")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expected.add(f"model.layers.0.mlp.{proj}")
        expected.add(f"model.layers.1.mlp.shared_mlp.{proj}")
    for path in expected:
        module = _resolve(model, path)
        assert isinstance(module, nn.QuantizedLinear), path
        assert module.bits == 8 and module.group_size == 64, path

    # Independent q8 -> q4 reference for one covered module, captured BEFORE
    # the pass replaces it.
    q8 = model.model.layers[1].self_attn.q_proj
    ref_w = mx.dequantize(
        q8.weight, q8.scales, q8.biases,
        group_size=q8.group_size, bits=q8.bits, mode=q8.mode,
    )
    ref_wq, ref_scales, ref_biases = mx.quantize(ref_w, 64, 4, mode="affine")

    requantized = _runtime_requantize_projections(model, "q4")
    assert set(requantized) == expected

    for path in expected:
        module = _resolve(model, path)
        assert isinstance(module, nn.QuantizedLinear), path
        assert module.bits == 4 and module.group_size == 64, path
        assert module.mode == "affine", path

    # The router gate, lm_head, embeddings, and norms keep loaded precision.
    router_gate = model.model.layers[1].mlp.router.gate
    assert isinstance(router_gate, nn.Linear)
    assert not isinstance(router_gate, nn.QuantizedLinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)
    assert isinstance(model.model.embed_tokens, nn.Embedding)
    assert isinstance(model.model.layers[1].input_layernorm, nn.RMSNorm)

    # BITWISE parity with the independent dequantize(q8) -> quantize(q4)
    # reference: from_linear runs the same mx.quantize path a load-time
    # quantization would, so the result is a standard q4 QuantizedLinear.
    new_qp = model.model.layers[1].self_attn.q_proj
    mx.eval(
        new_qp.weight, new_qp.scales, new_qp.biases,
        ref_wq, ref_scales, ref_biases,
    )
    assert bool(mx.array_equal(new_qp.weight, ref_wq).item())
    assert bool(mx.array_equal(new_qp.scales, ref_scales).item())
    assert bool(mx.array_equal(new_qp.biases, ref_biases).item())


def test_runtime_requantize_leaves_non_covered_quantized_modules_untouched() -> None:
    args = _tiny_args()
    model = Hy3Model(args)
    _prequantize_covered_to_q8(model)
    # Also quantize a NON-covered module (the router gate) to q8: it must be
    # left at q8 because proj_quant_covers excludes it, even though it is a
    # QuantizedLinear above the target bit width.
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        mode="affine",
        class_predicate=lambda path, module: (
            path == "model.layers.1.mlp.router.gate"
            and isinstance(module, nn.Linear)
            and not isinstance(module, nn.QuantizedLinear)
        ),
    )
    router_gate = model.model.layers[1].mlp.router.gate
    assert isinstance(router_gate, nn.QuantizedLinear) and router_gate.bits == 8

    _runtime_requantize_projections(model, "q4")

    still = model.model.layers[1].mlp.router.gate
    assert isinstance(still, nn.QuantizedLinear)
    assert still.bits == 8, "non-covered router gate must not be requantized"


def test_runtime_requantize_rejects_matchless_model() -> None:
    # A model with no covered quantized *_proj modules must fail loud, exactly
    # like proj_quant's matchless guard.
    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(8, 8, bias=False)

    with pytest.raises(ResidentLoadError, match=r"proj_requant.*matched no"):
        _runtime_requantize_projections(Bare(), "q4")


def test_runtime_requantize_is_idempotent_on_q4_residents() -> None:
    # Residents already at the target bit width are not touched a second time;
    # a fresh q4-only covered scope has nothing above the target -> raises.
    args = _tiny_args()
    model = Hy3Model(args)
    _runtime_quantize_projections(model, "q4")  # covered scope now q4/gs64
    with pytest.raises(ResidentLoadError, match=r"proj_requant.*matched no"):
        _runtime_requantize_projections(model, "q4")


def test_proj_requant_survives_benchmark_option_pipeline() -> None:
    """CLI vector -> parser -> runtime options -> config factory, mirroring the
    proj_quant plumbing test: the requant knob must not silently drop."""

    from types import SimpleNamespace

    bench = _load_benchmark_module()
    parser = bench.build_parser()
    assert bench.DEFAULT_RUNTIME_OPTIONS["proj_requant"] is None

    apis = SimpleNamespace(
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
    )

    args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--proj-requant", "q4",
        ]
    )
    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(args),
    }
    assert options["proj_requant"] == "q4"
    config = bench._runtime_config(apis, "hy3-expert-oq2e", options)
    assert config.proj_requant == "q4"
    # proj_requant must not force proj_quant on.
    assert config.proj_quant is None

    default_args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
        ]
    )
    default_options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(default_args),
    }
    assert default_options["proj_requant"] is None
    default_config = bench._runtime_config(apis, "hy3-expert-oq2e", default_options)
    assert default_config.proj_requant is None

    # q8 is rejected at the argparse boundary (choices=("q4",)).
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model", "hy3-q2",
                "--hy3-q2-model-root", "/tmp",
                "--proj-requant", "q8",
            ]
        )


def _resolve(model, path: str):
    module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def test_requant_plan_discount_halves_covered_u32_weights_only() -> None:
    from types import SimpleNamespace

    from mtplx.expert_runtime import proj_requant_plan_discount

    tensors = (
        SimpleNamespace(  # covered q8 packed weight -> halves
            tensor="model.layers.1.self_attn.q_proj.weight",
            dtype="U32",
            length=1_048_576,
        ),
        SimpleNamespace(  # scales keep their bytes (same group count)
            tensor="model.layers.1.self_attn.q_proj.scales",
            dtype="BF16",
            length=32_768,
        ),
        SimpleNamespace(  # shared_mlp covered too
            tensor="model.layers.1.mlp.shared_mlp.gate_proj.weight",
            dtype="U32",
            length=524_288,
        ),
        SimpleNamespace(  # router stays exact
            tensor="model.layers.1.mlp.router.gate.weight",
            dtype="BF16",
            length=1_048_576,
        ),
        SimpleNamespace(  # lm_head not covered
            tensor="lm_head.weight",
            dtype="U32",
            length=1_048_576,
        ),
        SimpleNamespace(  # BF16 residents untouched by requant
            tensor="model.layers.2.self_attn.q_proj.weight",
            dtype="BF16",
            length=1_048_576,
        ),
    )
    manifest = SimpleNamespace(resident_tensors=tensors)
    assert proj_requant_plan_discount(manifest, None) == 0
    assert proj_requant_plan_discount(manifest, "q4") == (
        1_048_576 // 2 + 524_288 // 2
    )
