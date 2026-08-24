from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.backends.dflash2 as backend
from mtplx.backends.dflash2 import (
    DEFAULT_DFLASH2_BLOCK_SIZE,
    DFlash2Runtime,
    DFlash2RuntimeConfig,
    DFlash2Unsupported,
    load_dflash2_bundle,
    resolve_dflash2_bundle_paths,
)
from mtplx.sampling import SamplerConfig
from tests.dflash2_test_bundle import write_exact_bundle


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return write_exact_bundle(tmp_path / "bundle", monkeypatch=monkeypatch)


def _isolate_turbo_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from mtplx.profiles import get_profile

    for name in get_profile("turbo").env_dict():
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("MTPLX_QWEN38_DISABLE_SOURCE_AUTO", "")


def _runtime(tmp_path: Path) -> DFlash2Runtime:
    target_runtime = SimpleNamespace(model="target", tokenizer=SimpleNamespace(
        decode=lambda ids: "".join(map(str, ids))
    ))
    return DFlash2Runtime(
        target_model="target",
        tokenizer=target_runtime.tokenizer,
        draft_model="draft",
        target_runtime=target_runtime,
        target_ops="ops",
        draft_backend="draft-backend",
        runtime_context="context",
        config=DFlash2RuntimeConfig.from_paths(
            target_model_path=tmp_path,
            draft_model_path=tmp_path,
        ),
    )


def test_resolver_uses_physical_width_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, monkeypatch)
    resolved = resolve_dflash2_bundle_paths(root)
    assert resolved is not None
    assert resolved["target_model"] == str(root / "target")
    assert resolved["draft_model"] == str(root / "dflash2")
    assert resolved["draft_quantization"] == "4bit"
    assert resolved["draft_block_size"] == DEFAULT_DFLASH2_BLOCK_SIZE == 8


def test_runtime_config_rejects_physical_width_outside_one_through_eight(
    tmp_path: Path,
) -> None:
    for width in (0, 9):
        config = DFlash2RuntimeConfig.from_paths(
            target_model_path=tmp_path,
            draft_model_path=tmp_path,
            draft_block_size=width,
        )
        with pytest.raises(ValueError, match="physical range 1..8"):
            config.validate_static()


def test_runtime_config_defaults_adaptive_and_accepts_fixed_opt_out(
    tmp_path: Path,
) -> None:
    adaptive = DFlash2RuntimeConfig.from_paths(
        target_model_path=tmp_path,
        draft_model_path=tmp_path,
    )
    fixed = DFlash2RuntimeConfig.from_paths(
        target_model_path=tmp_path,
        draft_model_path=tmp_path,
        draft_adaptive=False,
    )

    assert adaptive.draft_adaptive is True
    assert fixed.draft_adaptive is False


def test_load_uses_target_only_mtplx_and_pinned_dflash_mlx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass(frozen=True)
    class DraftCapabilities:
        default_block_tokens: int = 5
        max_block_tokens: int = 5

    root = _bundle(tmp_path, monkeypatch)
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "512")
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "50")
    _isolate_turbo_environment(monkeypatch)
    calls: list[tuple] = []
    target_runtime = SimpleNamespace(
        model=SimpleNamespace(), tokenizer=SimpleNamespace(decode=lambda ids: str(ids))
    )
    draft_model = SimpleNamespace(
        block_size=8,
        target_layer_ids=(5, 19, 33, 47, 61),
        capabilities=DraftCapabilities(),
    )
    import mtplx.runtime
    import dflash_mlx.runtime.loading
    import dflash_mlx.engine.target_qwen_gdn
    import dflash_mlx.engine.target_ops
    import dflash_mlx.runtime.context

    monkeypatch.setattr(
        mtplx.runtime,
        "load",
        lambda path, *, mtp: calls.append(("target", Path(path), mtp)) or target_runtime,
    )
    monkeypatch.setattr(
        dflash_mlx.runtime.loading,
        "load_draft_bundle",
        lambda path, **kwargs: (
            calls.append(("draft", Path(path), kwargs)) or (draft_model, {"pinned": True})
        ),
    )
    monkeypatch.setattr(
        dflash_mlx.engine.target_qwen_gdn.QwenGdnTargetOps,
        "supports_model",
        lambda self, model: True,
    )
    monkeypatch.setattr(
        dflash_mlx.engine.target_ops,
        "bind_draft_to_target",
        lambda draft, target, *, target_ops: calls.append(("bind", draft, target)),
    )
    monkeypatch.setattr(
        dflash_mlx.runtime.context,
        "build_offline_runtime_context",
        lambda **kwargs: {"context": kwargs},
    )
    monkeypatch.setattr(
        backend,
        "_install_measured_qwen38_dflash_stack",
        lambda runtime: {"retained": True},
    )

    runtime = load_dflash2_bundle(root)

    assert calls[0] == ("target", root / "target", False)
    assert calls[1][0:2] == ("draft", root / "dflash2")
    assert calls[1][2]["draft_quant"] == "w4:gs64"
    assert runtime.backend_id == "dflash2"
    assert runtime.config.draft_block_size == 8
    assert runtime.draft_model.capabilities.default_block_tokens == 8
    assert runtime.draft_model.capabilities.max_block_tokens == 8
    assert runtime.qwen38_feature_receipt == {"retained": True}
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "512"
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "50"
    assert os.environ["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert os.environ["MTPLX_GQA_PACKED_SDPA"] == "1"


def test_load_requires_tuned_row53_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, monkeypatch)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)

    with pytest.raises(RuntimeError, match="must be 512 and 50 before importing MLX"):
        load_dflash2_bundle(root)


def test_package_bootstrap_selects_tuned_row53_for_dflash2_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, monkeypatch)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    _isolate_turbo_environment(monkeypatch)
    import mtplx

    assert mtplx._bootstrap_dflash2_command_buffer_environment(
        ["mtplx", "serve", "--model", str(root)]
    ) is True
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "512"
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "50"
    assert os.environ["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert os.environ["MTPLX_GQA_PACKED_SDPA"] == "1"
    assert os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] == "1"


def test_measured_context_route_splits_only_the_supported_phase_controls() -> None:
    assert backend.qwen38_dflash_context_route(1024) == {
        "route_id": "short_lt16384",
        "adaptive_active": True,
        "row21_active": True,
        "row24_prefill_active": True,
        "row24_decode_active": True,
        "row48_prefill_active": True,
        "row48_decode_active": True,
        "row50_active": True,
        "requested_adaptive": True,
        "effective_adaptive": True,
        "fixed_block_size": None,
    }
    assert backend.qwen38_dflash_context_route(16_384) == {
        "route_id": "long_ge16384",
        "adaptive_active": True,
        "row21_active": True,
        "row24_prefill_active": True,
        "row24_decode_active": True,
        "row48_prefill_active": True,
        "row48_decode_active": True,
        "row50_active": True,
        "requested_adaptive": True,
        "effective_adaptive": True,
        "fixed_block_size": None,
    }
    assert backend.qwen38_dflash_context_route(
        131_072, adaptive_active=False
    ) == {
        "route_id": "long_ge16384",
        "adaptive_active": False,
        "row21_active": True,
        "row24_prefill_active": True,
        "row24_decode_active": True,
        "row48_prefill_active": True,
        "row48_decode_active": True,
        "row50_active": True,
        "requested_adaptive": False,
        "effective_adaptive": False,
        "fixed_block_size": 8,
    }


def test_context_route_is_applied_to_every_phase_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    runtime.qwen38_feature_receipt = {"installed": True}
    calls: dict[str, list[tuple]] = {
        "row21": [],
        "row24_qk": [],
        "row24": [],
        "row48": [],
        "adaptive": [],
        "row50": [],
    }
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row21_qk_rms_rope",
        lambda model, *, active: calls["row21"].append((active,)) or {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row24_qk_length_limit",
        lambda model, *, active, max_length: calls["row24_qk"].append(
            (active, max_length)
        )
        or {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_dflash_row24_eval_ladder",
        lambda model, **kwargs: calls["row24"].append(
            (kwargs["prefill_active"], kwargs["decode_active"])
        )
        or {"active": kwargs["active"]},
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.configure_qwen38_dflash_row48_boundary",
        lambda model, **kwargs: calls["row48"].append(
            (kwargs["prefill_active"], kwargs["decode_active"])
        )
        or {"active": kwargs["active"]},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_dflash_adaptive.configure_qwen38_dflash_adaptive_policy",
        lambda model, *, active, proposal_rows: calls["adaptive"].append((active,))
        or {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row50_wired_residency",
        lambda target_runtime, *, active: calls["row50"].append((active,))
        or {"installed": active},
    )

    for prompt_tokens in (1024, 16_384, 65_536, 131_072):
        backend._apply_measured_qwen38_dflash_context_route(
            runtime, prompt_tokens=prompt_tokens
        )

    assert calls == {
        "row21": [(True,)] * 4,
        "row24_qk": [(True, 32)] * 4,
        "row24": [(True, True)] * 4,
        "row48": [(True, True)] * 4,
        "adaptive": [(True,)] * 4,
        "row50": [(True,)] * 4,
    }
    assert runtime.qwen38_feature_receipt["context_route"]["route_id"] == (
        "long_ge16384"
    )
    assert runtime.qwen38_feature_receipt["context_route"]["requested_adaptive"] is True
    assert runtime.qwen38_feature_receipt["context_route"]["effective_adaptive"] is True
    assert runtime.qwen38_feature_receipt["context_route"]["fixed_block_size"] is None


def test_fixed_runtime_disables_adaptive_at_every_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    runtime.config = DFlash2RuntimeConfig.from_paths(
        target_model_path=tmp_path,
        draft_model_path=tmp_path,
        draft_adaptive=False,
    )
    runtime.qwen38_feature_receipt = {"installed": True}
    adaptive_calls: list[bool] = []
    monkeypatch.setattr(
        "mtplx.qwen38_dflash_adaptive.configure_qwen38_dflash_adaptive_policy",
        lambda model, *, active, proposal_rows: adaptive_calls.append(active)
        or {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row21_qk_rms_rope",
        lambda model, *, active: {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row24_qk_length_limit",
        lambda model, *, active, max_length: {"active": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_dflash_row24_eval_ladder",
        lambda model, **kwargs: {"active": kwargs["active"]},
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.configure_qwen38_dflash_row48_boundary",
        lambda model, **kwargs: {"active": kwargs["active"]},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row50_wired_residency",
        lambda target_runtime, *, active: {"installed": active},
    )

    for prompt_tokens in (1024, 16_384, 65_536, 131_072):
        backend._apply_measured_qwen38_dflash_context_route(
            runtime, prompt_tokens=prompt_tokens
        )

    assert adaptive_calls == [False] * 4
    assert runtime.qwen38_feature_receipt["context_route"]["fixed_block_size"] == 8


def test_measured_stack_installs_survivors_adaptive_and_releases_native_mtp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row21_qk_rms_rope",
        lambda model, *, active: {"installed": active},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_row24_qk_length_limit",
        lambda model, *, active, max_length: {"installed": active, "max_length": max_length},
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_dflash_row24_eval_ladder",
        lambda model, *, active, prefill_stride, prefill_active, decode_active: {
            "installed": active,
            "prefill_stride": prefill_stride,
            "prefill_active": prefill_active,
            "decode_active": decode_active,
        },
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_dflash_gqa_widths",
        lambda model, *, active, widths: {
            "active": active,
            "widths": list(widths),
        },
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge_kernels.configure_qwen38_dflash_m8_nax_island",
        lambda model, *, active, include_m8_output=True, include_m7_output=False, include_m7_linear_z=False, include_m8_kv=False, include_m8_qkv=False, include_m8_mlp=False, include_m5_exact=False, include_m6_kp1=False: {
            "active": active,
            "width": 8,
            "include_m8_output": include_m8_output,
            "include_m7_output": include_m7_output,
            "include_m7_linear_z": include_m7_linear_z,
            "include_m8_kv": include_m8_kv,
            "include_m8_qkv": include_m8_qkv,
            "include_m8_mlp": include_m8_mlp,
            "include_m5_exact": include_m5_exact,
            "include_m6_kp1": include_m6_kp1,
            "m6_kp1_shapes": (
                [[5120, 10240], [5120, 17408]] if include_m6_kp1 else []
            ),
            "shapes": (
                [[5120, 1024], [6144, 5120]]
                if include_m8_kv and not include_m8_qkv
                else [[5120, 1024], [5120, 10240], [5120, 17408]]
                if include_m8_mlp
                else [[5120, 1024], [5120, 10240]]
                if include_m8_qkv
                else ([] if not include_m8_output else [[6144, 5120]])
            ),
            "m7_shapes": (
                [[5120, 6144], [6144, 5120]]
                if include_m7_linear_z
                else ([[6144, 5120]] if include_m7_output else [])
            ),
            "eligible_projections": 16 if include_m8_output else 0,
            "eligible_m7_projections": (
                64 if include_m7_linear_z else (16 if include_m7_output else 0)
            ),
            "eligible_m7_linear_z_projections": 48 if include_m7_linear_z else 0,
            "m8_expanded_shapes": (
                [[5120, 1024], [5120, 10240], [5120, 17408]]
                if include_m8_mlp
                else [[5120, 1024], [5120, 10240]]
                if include_m8_qkv
                else ([[5120, 1024]] if include_m8_kv else [])
            ),
            "eligible_m8_expanded_projections": (
                192
                if include_m8_mlp
                else (80 if include_m8_qkv else (32 if include_m8_kv else 0))
            ),
        },
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.configure_qwen38_dflash_row48_boundary",
        lambda model, *, active, prefill_active, decode_active: {
            "installed": active,
            "prefill_active": prefill_active,
            "decode_active": decode_active,
        },
    )
    monkeypatch.setattr(
        "mtplx.qwen38_dflash_adaptive.configure_qwen38_dflash_adaptive_policy",
        lambda model, *, active, proposal_rows: {
            "active": active, "proposal_rows": list(proposal_rows),
            "min_block_tokens": 1, "max_block_tokens": 8,
        },
    )
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row50_wired_residency",
        lambda target_runtime, *, active: {"installed": active},
    )
    monkeypatch.setattr(
        "mtplx.nax_verify.configure_qwen38_m6_barrier_free_kp1",
        lambda *, active: {"active": active},
    )

    receipt = backend._install_measured_qwen38_dflash_stack(runtime)

    assert receipt["r24_qk_length_limit"]["max_length"] == 32
    assert receipt["r24_eval_ladder"]["prefill_stride"] == 3
    assert receipt["adaptive_policy"]["proposal_rows"] == [11, 15]
    assert receipt["dflash_m8_nax_island"] == {
        "active": True,
        "width": 8,
        "include_m8_output": False,
        "include_m7_output": True,
        "include_m7_linear_z": True,
        "include_m8_kv": True,
        "include_m8_qkv": True,
        "include_m8_mlp": True,
        "include_m5_exact": True,
        "include_m6_kp1": True,
        "m6_kp1_shapes": [[5120, 10240], [5120, 17408]],
        "shapes": [[5120, 1024], [5120, 10240], [5120, 17408]],
        "m7_shapes": [[5120, 6144], [6144, 5120]],
        "eligible_projections": 0,
        "eligible_m7_projections": 64,
        "eligible_m7_linear_z_projections": 48,
        "m8_expanded_shapes": [[5120, 1024], [5120, 10240], [5120, 17408]],
        "eligible_m8_expanded_projections": 192,
    }
    assert receipt["native_mtp_release"] == {
        "native_mtp_released": True,
        "native_mtp_loaded": False,
    }
    assert receipt["dflash_m6_barrier_free_kp1"] == {"active": True}


def test_dflash_mlx_events_map_to_generation_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dflash_mlx.engine.events import CycleCompleteEvent, SummaryEvent, TokenEvent
    import dflash_mlx.runtime

    runtime = _runtime(tmp_path)
    cycle = CycleCompleteEvent(
        cycle=1, block_len=8, commit_count=3, acceptance_len=2,
        draft_us=1, verify_us=2, acceptance_us=1, hidden_extraction_us=1,
        rollback_us=0, other_us=1, cycle_total_us=6,
    )
    summary = SummaryEvent(
        elapsed_us=2_000_000, prompt_token_count=2,
        generated_token_ids=(11, 12, 13), generation_tokens=3,
        accepted_from_draft=2, acceptance_ratio=2 / 3, cycles_completed=1,
        phase_timings_us={"prefill": 500_000}, block_tokens=8,
        peak_memory_gb=1.5,
        adaptive_metrics={"policy": "qwen38_position_ema", "max_block_tokens": 8},
    )
    monkeypatch.setattr(
        dflash_mlx.runtime,
        "stream_dflash_generate",
        lambda **kwargs: iter((TokenEvent(11, 1, 1.0, 1), cycle, summary)),
    )

    @contextmanager
    def sampling(*args, **kwargs):
        yield

    monkeypatch.setattr(backend, "_dflash_target_sampling", sampling)
    chunks: list[list[int]] = []
    output = backend.generate_dflash2(
        runtime,
        [1, 2],
        max_tokens=3,
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
        token_callback=chunks.append,
    )
    assert output.tokens == [11, 12, 13]
    assert chunks == [[11]]
    assert output.stats.accepted_drafts == 2
    assert output.stats.drafted_tokens == 7
    assert output.stats.prompt_tps == 4.0
    assert output.stats.decode_tok_s == 2.0
    assert output.stats.tok_s == 2.0
    assert output.stats.end_to_end_tok_s == 1.5
    assert output.stats.peak_memory_bytes == 1_500_000_000
    assert output.stats.draft_core["adaptive_metrics"] == {
        "policy": "qwen38_position_ema",
        "max_block_tokens": 8,
    }


def test_summary_block_histogram_accounts_for_drafts_without_cycle_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dflash_mlx.engine.events import SummaryEvent
    import dflash_mlx.runtime

    runtime = _runtime(tmp_path)
    summary = SummaryEvent(
        elapsed_us=2_000_000,
        prompt_token_count=2,
        generated_token_ids=(11, 12, 13),
        generation_tokens=3,
        accepted_from_draft=2,
        acceptance_ratio=2 / 3,
        cycles_completed=4,
        phase_timings_us={"prefill": 500_000},
        block_tokens=8,
        adaptive_metrics={"cycles_by_block": {"2": 1, "4": 3}},
    )
    monkeypatch.setattr(
        dflash_mlx.runtime,
        "stream_dflash_generate",
        lambda **kwargs: iter((summary,)),
    )

    @contextmanager
    def sampling(*args, **kwargs):
        yield

    monkeypatch.setattr(backend, "_dflash_target_sampling", sampling)
    output = backend.generate_dflash2(
        runtime,
        [1, 2],
        max_tokens=3,
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
    )

    assert output.stats.events == []
    assert output.stats.drafted_tokens == 10


def test_generation_rejects_unsupported_sessions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(DFlash2Unsupported, match="sessions"):
        backend.generate_dflash2(
            runtime,
            [1],
            max_tokens=2,
            sampler=SamplerConfig(temperature=0.0),
            session_bank=object(),
        )
