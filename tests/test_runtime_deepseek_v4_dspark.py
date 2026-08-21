from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_dspark_artifact import DSparkConfig  # noqa: E402
from mtplx.deepseek_v4_dspark_runtime import (  # noqa: E402
    DeepseekV4DSparkRuntime,
    install_deepseek_v4_dspark_runtime,
)
from mtplx.models.deepseek_v4 import DeepseekV4AffineInt4Cache  # noqa: E402
from mtplx.models.deepseek_v4_dspark import DeepseekV4DSparkCache  # noqa: E402
from mtplx import runtime  # noqa: E402


def test_installed_runtime_binds_only_fixed_k5_m6_and_affine_int4_caches() -> None:
    target_caches = [
        DeepseekV4AffineInt4Cache(window_size=128, compress_ratio=0, head_dim=512)
    ]
    draft_caches = [
        DeepseekV4DSparkCache(window_size=128, head_dim=512) for _ in range(3)
    ]
    calls = []

    class _Model:
        dspark = SimpleNamespace(stages=[object(), object(), object()])

        def make_cache(self):
            return target_caches

        def make_dspark_cache(self):
            return draft_caches

        def __call__(self, input_ids, *, cache, return_hidden):
            calls.append(("target_m6", int(input_ids.shape[1]), cache))
            logits = mx.zeros((1, 6, 64))
            taps = tuple(mx.zeros((1, 6, 2)) for _ in range(3))
            return logits, taps

        def propose_dspark_k5(self, primary, caches, *, start_pos):
            calls.append(("proposal_k5", int(primary.item()), caches, start_pos))
            return SimpleNamespace(
                future_tokens=mx.array([[31, 32, 33, 34, 35]], dtype=mx.int32)
            )

        def commit_dspark_main(self, taps, caches, *, start_pos):
            calls.append(("commit_dspark", taps, caches, start_pos))

    artifact = SimpleNamespace(
        config=DSparkConfig(
            block_size=5,
            markov_rank=256,
            noise_token_id=128799,
            target_layer_ids=(40, 41, 42),
            stage_ids=(0, 1, 2),
        )
    )
    installed = install_deepseek_v4_dspark_runtime(_Model(), artifact)

    assert isinstance(installed, DeepseekV4DSparkRuntime)
    assert installed.make_target_cache() is target_caches
    assert installed.make_dspark_cache() is draft_caches
    verified = installed.target_m6(
        mx.array([[29, 31, 32, 33, 34, 35]], dtype=mx.int32),
        target_caches,
    )
    proposed = installed.proposal_k5(
        mx.array([29], dtype=mx.int32),
        draft_caches,
        9,
    )

    assert tuple(verified.logits.shape) == (1, 6, 64)
    assert tuple(proposed.shape) == (1, 5)
    assert calls[:2] == [
        ("target_m6", 6, target_caches),
        ("proposal_k5", 29, draft_caches, 9),
    ]
    assert not hasattr(installed, "mtp_forward")
    with pytest.raises(FrozenInstanceError):
        installed.config = None


def test_explicit_load_qualifies_artifact_before_model_and_skips_generic_mtp(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    config = {
        "model_type": "deepseek_v4",
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
    }
    artifact = SimpleNamespace(
        config=DSparkConfig(5, 256, 128799, (40, 41, 42), (0, 1, 2))
    )

    class _LoadModel:
        dspark = SimpleNamespace(stages=[object(), object(), object()])

        def make_cache(self):
            return [DeepseekV4AffineInt4Cache(128, 0, 512)]

        def make_dspark_cache(self):
            return [DeepseekV4DSparkCache(window_size=128, head_dim=512) for _ in range(3)]

    model = _LoadModel()
    monkeypatch.setattr(runtime, "load_config", lambda _path: config)
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(
        "mtplx.deepseek_v4_dspark_artifact.open_verified_dspark_artifact",
        lambda _path: events.append("artifact") or artifact,
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: events.append("model") or (model, object()),
    )

    import mtplx.a3b_compiled_target_prefix as target_prefix
    import mtplx.a3b_whole_moe as whole_moe
    import mtplx.attention_split as attention_split
    import mtplx.gdn_capture as gdn_capture
    import mtplx.kernel_selfcheck as kernel_selfcheck
    import mtplx.native_mlp as native_mlp
    import mtplx.nax_verify as nax_verify
    import mtplx.qwen_row_owned_router as row_owned

    monkeypatch.setattr(attention_split, "configure_split_full_attention", lambda *_: None)
    monkeypatch.setattr(native_mlp, "configure_native_mlp", lambda *_: None)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", lambda: False)
    monkeypatch.setattr(whole_moe, "prepare_a3b_whole_moe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(row_owned, "prepare_qwen_row_owned_routers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gdn_capture, "prepare_a3b_gdn_postconv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", lambda *_: None)
    monkeypatch.setattr(target_prefix, "prepare_a3b_compiled_target_prefix", lambda *_args, **_kwargs: None)

    loaded = runtime.load(tmp_path, mtp=True, dspark=True)

    assert events[:2] == ["artifact", "model"]
    assert loaded.mtp_enabled is False
    assert isinstance(loaded.deepseek_v4_dspark_runtime, DeepseekV4DSparkRuntime)


def test_server_parser_accepts_only_explicit_fixed_dspark_route() -> None:
    from mtplx.server.openai import parse_args

    args = parse_args(
        [
            "--model",
            "/tmp/dspark-model",
            "--generation-mode",
            "dspark",
            "--depth",
            "5",
            "--temperature",
            "0",
            "--warmup-tokens",
            "0",
        ]
    )

    assert args.generation_mode == "dspark"
    assert args.depth == 5
    assert args.temperature == 0.0
