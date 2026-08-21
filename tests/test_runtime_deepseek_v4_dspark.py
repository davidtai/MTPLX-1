from types import SimpleNamespace

import pytest

from mtplx import runtime
from mtplx.deepseek_v4_dspark_artifact import DSparkConfig
from mtplx.models.deepseek_v4 import DeepseekV4AffineInt4Cache
from mtplx.models.deepseek_v4_dspark import DeepseekV4DSparkCache


def test_explicit_load_qualifies_dspark_without_installing_a_second_runtime(
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
            return [
                DeepseekV4DSparkCache(window_size=128, head_dim=512)
                for _ in range(3)
            ]

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
    assert loaded.model is model
    assert loaded.mtp_enabled is False
    assert not hasattr(loaded, "deepseek_v4_dspark_runtime")


def test_server_parser_accepts_explicit_fixed_dspark_route() -> None:
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


def test_server_dspark_route_requires_the_bound_dflash2_bundle() -> None:
    from fastapi import HTTPException
    from mtplx.server.openai import _request_generation_mode_for_generation

    request = SimpleNamespace(generation_mode=None, model_extra=None)
    state = SimpleNamespace(
        args=SimpleNamespace(generation_mode="dspark"),
        runtime=SimpleNamespace(mtp_enabled=False),
        deepseek_v4_dflash2_bundle=object(),
    )

    assert _request_generation_mode_for_generation(state, request) == "dspark"
    state.deepseek_v4_dflash2_bundle = None
    with pytest.raises(HTTPException, match="DFlash2-qualified"):
        _request_generation_mode_for_generation(state, request)
