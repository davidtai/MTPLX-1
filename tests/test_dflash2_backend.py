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


def test_load_rejects_row53_environment_that_was_not_bootstrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, monkeypatch)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)

    with pytest.raises(RuntimeError, match="must be set before importing MLX"):
        load_dflash2_bundle(root)


def test_package_bootstrap_latches_row53_for_dflash2_model(
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
        lambda model, *, active, prefill_stride: {
            "installed": active, "prefill_stride": prefill_stride
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
        lambda model, *, active, include_m7_output=False, include_m7_linear_z=False, include_m8_kv=False: {
            "active": active,
            "width": 8,
            "include_m7_output": include_m7_output,
            "include_m7_linear_z": include_m7_linear_z,
            "include_m8_kv": include_m8_kv,
            "shapes": (
                [[5120, 1024], [6144, 5120]]
                if include_m8_kv
                else [[6144, 5120]]
            ),
            "m7_shapes": (
                [[5120, 6144], [6144, 5120]]
                if include_m7_linear_z
                else ([[6144, 5120]] if include_m7_output else [])
            ),
            "eligible_projections": 16,
            "eligible_m7_projections": (
                64 if include_m7_linear_z else (16 if include_m7_output else 0)
            ),
            "eligible_m7_linear_z_projections": 48 if include_m7_linear_z else 0,
            "m8_expanded_shapes": [[5120, 1024]] if include_m8_kv else [],
            "eligible_m8_expanded_projections": 32 if include_m8_kv else 0,
        },
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.configure_qwen38_dflash_row48_boundary",
        lambda model, *, active: {"installed": active},
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

    receipt = backend._install_measured_qwen38_dflash_stack(runtime)

    assert receipt["r24_qk_length_limit"]["max_length"] == 32
    assert receipt["r24_eval_ladder"]["prefill_stride"] == 3
    assert receipt["adaptive_policy"]["proposal_rows"] == [11, 15]
    assert receipt["dflash_m8_nax_island"] == {
        "active": True,
        "width": 8,
        "include_m7_output": True,
        "include_m7_linear_z": True,
        "include_m8_kv": True,
        "shapes": [[5120, 1024], [6144, 5120]],
        "m7_shapes": [[5120, 6144], [6144, 5120]],
        "eligible_projections": 16,
        "eligible_m7_projections": 64,
        "eligible_m7_linear_z_projections": 48,
        "m8_expanded_shapes": [[5120, 1024]],
        "eligible_m8_expanded_projections": 32,
    }
    assert receipt["native_mtp_release"] == {
        "native_mtp_released": True,
        "native_mtp_loaded": False,
    }


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
