from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_cli import (
    add_expert_streaming_args,
    append_expert_streaming_child_args,
    expert_streaming_load_kwargs,
)
from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.expert_streaming_models import HY3_Q4
from mtplx.attention_context import attention_phase
from mtplx.expert_streaming import RoutingPhase
from mtplx.memory_broker import BINARY_GIB
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.runtime_options import validate_hy3_q4_dynamic_memory_options


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser)
    return parser


def _model_root(tmp_path: Path, model_type: str = "hy_v3") -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    (root / "expert-manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_dynamic_cache_is_direct_machine_configurable_and_record_granular() -> None:
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=100 * BINARY_GIB,
        max_live_kv_tokens=131_072,
        runtime_reserve_bytes=8 * BINARY_GIB,
        transient_slots=32,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="direct-slots",
        dynamic_expert_cache=True,
    )

    plan = config.memory_plan(HY3_Q4)

    assert config.memory_limit_bytes == 100 * BINARY_GIB
    assert plan.persistent_slots > 0
    assert plan.persistent_cache_bytes % HY3_Q4.expert_record_bytes == 0
    assert not hasattr(config, "dynamic_expert_slabs")
    assert not hasattr(config, "expert_slab_slots")


def test_dynamic_memory_cli_builds_direct_cache_config(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser, include_hy3_dynamic_memory=True)
    args = parser.parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "100GiB",
            "--expert-max-live-kv-tokens",
            "131072",
            "--expert-cache-scope",
            "global",
            "--expert-slot-layout",
            "direct-slots",
            "--hy3-q4-dynamic-memory",
        ]
    )

    config = expert_streaming_load_kwargs(args, root)["expert_streaming_config"]

    assert config.dynamic_expert_cache is True
    assert config.memory_limit_bytes == 100 * BINARY_GIB
    assert config.slot_layout == "direct-slots"
    assert not hasattr(config, "dynamic_expert_slabs")


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--expert-slab-slots",
        "--expert-regrow-hysteresis-slabs",
        "--expert-resize-min-interval-ms",
    ],
)
def test_dynamic_memory_cli_rejects_removed_slab_flags(removed_flag: str) -> None:
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser, include_hy3_dynamic_memory=True)

    with pytest.raises(SystemExit):
        parser.parse_args(["--hy3-q4-dynamic-memory", removed_flag, "1"])


def test_dynamic_memory_attestation_uses_direct_cache_capability() -> None:
    args = SimpleNamespace(
        hy3_q4_dynamic_memory=True,
        hy3_q4_dynamic_context=True,
        expert_streaming=True,
        expert_streaming_config=None,
        expert_manifest=None,
    )
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=100 * BINARY_GIB,
        max_live_kv_tokens=131_072,
        kv_bytes_per_token_override=84_480,
        runtime_reserve_bytes=8 * BINARY_GIB,
        allocator_headroom_bytes=BINARY_GIB,
        transient_slots=32,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="direct-slots",
        resource_telemetry=True,
        dynamic_expert_cache=True,
    )

    assert validate_hy3_q4_dynamic_memory_options(args, config) is True


def test_expert_cli_builds_explicit_bounded_config(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
            "--expert-cache-limit",
            "24GiB",
            "--expert-runtime-reserve",
            "8GiB",
            "--expert-allocator-headroom",
            "1GiB",
            "--no-expert-prefer-sidecar",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    config = kwargs["expert_streaming_config"]

    assert kwargs["mtp"] is False
    assert kwargs["expert_manifest"] == root / "expert-manifest.json"
    assert config.model_key == "hy3-q4"
    assert config.memory_limit_bytes == 96 * 1024**3
    assert config.max_live_kv_tokens == 8192
    assert config.expert_cache_limit_bytes == 24 * 1024**3
    assert config.runtime_reserve_bytes == 8 * 1024**3
    assert config.allocator_headroom_bytes == 1024**3
    assert config.prefer_sidecar is False


def test_expert_cli_json_and_flags_are_strict_and_forwarded(tmp_path: Path) -> None:
    root = _model_root(tmp_path, "glm_moe_dsa")
    config_path = tmp_path / "stream.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "glm52-q4",
                "memory_limit_bytes": "256GiB",
                "max_live_kv_tokens": 4096,
                "runtime_reserve_bytes": "12GiB",
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--expert-streaming-config",
            str(config_path),
            "--expert-memory-limit",
            "320GiB",
            "--no-expert-verify-record-hashes",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    assert kwargs["expert_streaming_config"].memory_limit_bytes == 320 * 1024**3
    assert kwargs["expert_streaming_config"].verify_record_hashes is False

    command = ["python", "-m", "mtplx.server.openai"]
    append_expert_streaming_child_args(command, args)
    assert "--expert-streaming" in command
    assert command[command.index("--expert-memory-limit") + 1] == "320GiB"
    assert "--no-expert-verify-record-hashes" in command

    args.expert_allocator_headroom = "2GiB"
    command = []
    append_expert_streaming_child_args(command, args)
    assert command[command.index("--expert-allocator-headroom") + 1] == "2GiB"


def test_expert_cli_requires_memory_and_kv_limits(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    args = _parser().parse_args(["--expert-streaming"])
    with pytest.raises(
        ValueError, match="missing memory-limit-bytes, max-live-kv-tokens"
    ):
        expert_streaming_load_kwargs(args, root)


def test_dynamic_memory_opt_in_is_forwarded_even_when_streaming_is_incomplete() -> None:
    args = SimpleNamespace(
        expert_streaming=False,
        expert_streaming_config=None,
        expert_manifest=None,
        hy3_q4_dynamic_memory=True,
    )
    command: list[str] = []

    append_expert_streaming_child_args(command, args)

    assert command == ["--expert-streaming", "--hy3-q4-dynamic-memory"]


class _PhaseModel:
    def __init__(self) -> None:
        self.phases: list[RoutingPhase] = []

    def __call__(self, input_ids, cache=None):
        del cache
        self.phases.append(current_expert_routing_phase(token_count=999))
        return input_ids


class _StreamingStub:
    def __init__(self) -> None:
        self.closed = False
        self.memory_broker = None

    def close(self, *, timeout=None) -> None:
        del timeout
        self.closed = True

    def snapshot(self):
        return {"ok": True}

    def admit_kv_tokens(self, tokens):
        return SimpleNamespace(tokens=tokens)


def test_mtplx_runtime_marks_prefill_decode_and_closes_streaming_runtime() -> None:
    model = _PhaseModel()
    streaming = _StreamingStub()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    runtime.forward_ar(SimpleNamespace(shape=(1, 3)))
    runtime.forward_ar(SimpleNamespace(shape=(1, 1)))

    assert model.phases == [RoutingPhase.PREFILL, RoutingPhase.DECODE]
    assert runtime.expert_streaming_snapshot() == {"ok": True}
    runtime.close()
    assert streaming.closed is True


def test_attention_phase_context_overrides_routing_shape_heuristic() -> None:
    model = _PhaseModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=_StreamingStub(),
    )

    # A one-token prefill tail chunk is still prefill traffic.
    with attention_phase("prefill"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))
    # MTP verify batches are decode traffic despite their width.
    with attention_phase("decode_verify"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 2)))
    with attention_phase("ar_decode"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))
    with attention_phase("postcommit"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 2)))
    # Unrecognized phases normalize to unknown and keep the heuristic.
    with attention_phase("ar_batch_shared_prefill"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 3)))
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))

    assert model.phases == [
        RoutingPhase.PREFILL,
        RoutingPhase.DECODE,
        RoutingPhase.DECODE,
        RoutingPhase.DECODE,
        RoutingPhase.PREFILL,
        RoutingPhase.DECODE,
    ]
    runtime.close()


def test_brokered_forward_prepares_all_q4_pages_before_model_execution(
    monkeypatch,
) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")

    events: list[object] = []

    class Model:
        def __call__(self, input_ids, cache=None):
            events.append(("model", cache))
            return input_ids

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    cache = [object()]

    def prepare(actual_cache, *, append_tokens):
        events.append(("prepare", actual_cache, append_tokens))
        return {"entries": 80}

    monkeypatch.setattr(cache_state, "prepare_brokered_q4_cache_group", prepare)

    runtime.forward_ar(SimpleNamespace(shape=(1, 7)), cache=cache)

    assert events == [("prepare", cache, 7), ("model", cache)]


@pytest.mark.parametrize(
    "preparation_result",
    [
        pytest.param(None, id="malformed"),
        pytest.param({"entries": 0}, id="zero"),
        pytest.param({"entries": 79}, id="short"),
        pytest.param({"entries": 81}, id="extra"),
    ],
)
def test_brokered_forward_rejects_incomplete_q4_group_before_model_execution(
    monkeypatch,
    preparation_result,
) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    model_calls: list[object] = []

    class Model:
        def __call__(self, input_ids, cache=None):
            model_calls.append((input_ids, cache))
            return input_ids

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    cache = [object()]
    monkeypatch.setattr(
        cache_state,
        "prepare_brokered_q4_cache_group",
        lambda _cache, *, append_tokens: preparation_result,
    )

    with pytest.raises(RuntimeError, match="aggregate Q4 cache attestation"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 7)), cache=cache)

    assert model_calls == []


def test_brokered_forward_keeps_non_dynamic_programmatic_path_unchanged(
    monkeypatch,
) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "0")
    calls: list[object] = []

    class Model:
        def __call__(self, input_ids, cache=None):
            calls.append((input_ids, cache))
            return input_ids

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    monkeypatch.setattr(
        cache_state,
        "prepare_brokered_q4_cache_group",
        lambda *_args, **_kwargs: pytest.fail("non-dynamic path prepared Q4 pages"),
    )
    input_ids = SimpleNamespace(shape=(1, 7))
    cache = [object()]

    assert runtime.forward_ar(input_ids, cache=cache) is input_ids
    assert calls == [(input_ids, cache)]


def test_brokered_capture_prepares_all_q4_pages_before_model_execution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")

    events: list[object] = []

    class Model:
        pass

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    cache = [object()]

    def prepare(input_ids, actual_cache):
        events.append(("prepare", input_ids, actual_cache))

    monkeypatch.setattr(runtime, "prepare_ar_cache_growth", prepare)
    monkeypatch.setattr(
        "mtplx.gdn_capture.forward_with_gdn_capture",
        lambda model, input_ids, **kwargs: events.append(
            ("capture", model, input_ids, kwargs)
        ),
    )
    input_ids = SimpleNamespace(shape=(1, 7))

    runtime.forward_ar_capture(input_ids, cache=cache)

    assert events == [
        ("prepare", input_ids, cache),
        (
            "capture",
            runtime.model,
            input_ids,
            {
                "cache": cache,
                "return_hidden": False,
                "hidden_variant": None,
                "capture_backend": None,
            },
        ),
    ]


def test_make_cache_aggregates_initial_q4_pages_before_return(monkeypatch) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")

    events: list[object] = []
    raw_cache = [object()]

    class Model:
        def make_cache(self):
            events.append("make")
            return raw_cache

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    monkeypatch.setattr(
        cache_state,
        "configure_owned_recurrent_state_cache",
        lambda cache: events.append(("recurrent", cache)),
    )
    monkeypatch.setattr(
        cache_state,
        "configure_tail_owned_attention_kv_cache",
        lambda cache, **kwargs: events.append(("configure", cache, kwargs)),
    )

    def prepare(cache):
        events.append(("prepare", cache))
        return {"entries": 80}

    monkeypatch.setattr(cache_state, "prepare_brokered_q4_cache_group", prepare)
    monkeypatch.setattr(
        cache_state,
        "register_physical_kv_cache",
        lambda cache: events.append(("register", cache)),
    )

    assert runtime.make_cache() is raw_cache
    assert events == [
        "make",
        ("recurrent", raw_cache),
        (
            "configure",
            raw_cache,
            {
                "allocation_observer": streaming,
                "cache_id_prefix": "target:0",
            },
        ),
        ("prepare", raw_cache),
        ("register", raw_cache),
    ]


@pytest.mark.parametrize(
    "preparation_result",
    [
        pytest.param(None, id="malformed"),
        pytest.param({"entries": 0}, id="zero"),
        pytest.param({"entries": 79}, id="short"),
        pytest.param({"entries": 81}, id="extra"),
    ],
)
def test_make_cache_closes_partial_owners_when_q4_group_attestation_fails(
    monkeypatch,
    preparation_result,
) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    raw_cache = [object()]
    close_calls: list[object] = []
    register_calls: list[object] = []

    class Model:
        @staticmethod
        def make_cache():
            return raw_cache

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    monkeypatch.setattr(
        cache_state,
        "configure_owned_recurrent_state_cache",
        lambda _cache: None,
    )
    monkeypatch.setattr(
        cache_state,
        "configure_tail_owned_attention_kv_cache",
        lambda _cache, **_kwargs: None,
    )
    monkeypatch.setattr(
        cache_state,
        "prepare_brokered_q4_cache_group",
        lambda _cache: preparation_result,
    )
    monkeypatch.setattr(
        cache_state,
        "close_physical_kv_cache",
        close_calls.append,
    )
    monkeypatch.setattr(
        cache_state,
        "register_physical_kv_cache",
        register_calls.append,
    )

    with pytest.raises(RuntimeError, match="aggregate Q4 cache attestation"):
        runtime.make_cache()

    assert close_calls == [raw_cache]
    assert register_calls == []


def test_make_cache_closes_partial_group_owners_when_preparation_fails(
    monkeypatch,
) -> None:
    import mtplx.cache_state as cache_state

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    events: list[object] = []
    raw_cache = [object()]

    class Model:
        @staticmethod
        def make_cache():
            return raw_cache

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    monkeypatch.setattr(
        cache_state,
        "configure_owned_recurrent_state_cache",
        lambda _cache: None,
    )
    monkeypatch.setattr(
        cache_state,
        "configure_tail_owned_attention_kv_cache",
        lambda _cache, **_kwargs: None,
    )

    def fail_prepare(_cache):
        events.append("prepare_failed")
        raise RuntimeError("injected group preparation failure")

    monkeypatch.setattr(
        cache_state,
        "prepare_brokered_q4_cache_group",
        fail_prepare,
    )
    monkeypatch.setattr(
        cache_state,
        "close_physical_kv_cache",
        lambda cache: events.append(("close", cache)),
    )

    with pytest.raises(RuntimeError, match="group preparation failure"):
        runtime.make_cache()

    assert events == ["prepare_failed", ("close", raw_cache)]


def test_dynamic_q4_runtime_rejects_unqualified_mtp_kv_cache(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")

    class Model:
        @staticmethod
        def make_mtp_cache():
            raise AssertionError("unqualified MTP cache construction was reached")

    streaming = _StreamingStub()
    streaming.memory_broker = object()
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=True,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    with pytest.raises(RuntimeError, match="target AR caches only"):
        runtime.make_mtp_cache()


class _MTPExecutionModel:
    def __init__(self) -> None:
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
        self.calls: list[str] = []

    def mtp_forward(self, *_args, **_kwargs):
        self.calls.append("draft")
        return "draft-result"

    def mtp_update_cache(self, *_args, **_kwargs):
        self.calls.append("update")
        return "update-result"


def _mtp_execution_runtime(model, *, memory_broker) -> MTPLXRuntime:
    streaming = _StreamingStub()
    streaming.memory_broker = memory_broker
    return MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=True,
        contract=MTPContract(),
        expert_streaming=streaming,
    )


def test_dynamic_q4_runtime_rejects_unqualified_mtp_draft_execution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    model = _MTPExecutionModel()
    runtime = _mtp_execution_runtime(model, memory_broker=object())

    with pytest.raises(RuntimeError, match="target AR caches only"):
        runtime.draft_mtp(None, None, mtp_cache=None)

    assert model.calls == []


def test_dynamic_q4_runtime_rejects_unqualified_mtp_cache_update(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    model = _MTPExecutionModel()
    runtime = _mtp_execution_runtime(model, memory_broker=object())

    with pytest.raises(RuntimeError, match="target AR caches only"):
        runtime.update_mtp_cache(None, None, mtp_cache=object())

    assert model.calls == []


def test_brokered_mtp_execution_remains_available_without_dynamic_paged_kv(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    model = _MTPExecutionModel()
    runtime = _mtp_execution_runtime(model, memory_broker=object())

    assert runtime.draft_mtp(None, None, mtp_cache=None) == "draft-result"
    assert runtime.update_mtp_cache(None, None, mtp_cache=object()) == "update-result"
    assert model.calls == ["draft", "update"]


def test_dynamic_paged_kv_without_live_broker_keeps_mtp_execution_available(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    model = _MTPExecutionModel()
    runtime = _mtp_execution_runtime(model, memory_broker=None)

    assert runtime.draft_mtp(None, None, mtp_cache=None) == "draft-result"
    assert runtime.update_mtp_cache(None, None, mtp_cache=object()) == "update-result"
    assert model.calls == ["draft", "update"]


def _patch_streamed_load_to_live_broker(
    monkeypatch,
    *,
    resident_boundary,
):
    import mtplx.expert_manifest as expert_manifest_module
    import mtplx.expert_runtime as expert_runtime_module
    import mtplx.models.expert_mlx as expert_mlx_module
    import mtplx.resident_loader as resident_loader_module

    runtime = SimpleNamespace(
        memory_broker=object(),
        closed=False,
    )

    def close(*, timeout=None):
        del timeout
        runtime.closed = True

    runtime.close = close
    monkeypatch.setattr(
        expert_manifest_module,
        "load_expert_manifest",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        expert_mlx_module,
        "make_mlx_slot_buffer_allocator",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        expert_runtime_module.ExpertStreamingRuntime,
        "open",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(
        resident_loader_module,
        "construct_resident_model",
        resident_boundary,
    )
    return runtime


def test_programmatic_load_preserves_static_component_bank_allocator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import mtplx.models.expert_mlx as expert_mlx_module
    from mtplx.runtime import load

    class ResidentLoadBoundary(RuntimeError):
        pass

    calls: list[tuple[object, ...]] = []

    def resident_boundary(*_args, **_kwargs):
        raise ResidentLoadBoundary

    _patch_streamed_load_to_live_broker(
        monkeypatch,
        resident_boundary=resident_boundary,
    )
    monkeypatch.setattr(
        expert_mlx_module,
        "make_mlx_component_bank_allocator",
        lambda *args, **kwargs: calls.append((*args, kwargs)) or object(),
    )
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=16 * BINARY_GIB,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    )

    with pytest.raises(ResidentLoadBoundary):
        load(
            tmp_path,
            mtp=False,
            expert_streaming_config=config,
            expert_manifest=tmp_path / "expert-manifest.json",
            model_config={"model_type": "hy_v3"},
        )

    assert len(calls) == 1
    assert calls[0][-1] == {}


def test_programmatic_load_rejects_mtp_when_strict_dynamic_q4_is_live(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mtplx.expert_runtime import ExpertStreamingConfig
    from mtplx.runtime import load

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")

    def resident_boundary(*_args, **_kwargs):
        raise AssertionError("resident model load reached after strict MTP rejection")

    expert_runtime = _patch_streamed_load_to_live_broker(
        monkeypatch,
        resident_boundary=resident_boundary,
    )
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )

    with pytest.raises(RuntimeError, match="target AR caches only"):
        load(
            tmp_path,
            mtp=True,
            expert_streaming_config=config,
            expert_manifest=tmp_path / "expert-manifest.json",
            model_config={"model_type": "hy_v3"},
            mtp_artifacts=tmp_path,
        )

    assert expert_runtime.closed is True


def test_programmatic_load_does_not_reject_brokered_mtp_when_dynamic_kv_is_off(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mtplx.expert_runtime import ExpertStreamingConfig
    from mtplx.runtime import load

    class ResidentLoadBoundary(RuntimeError):
        pass

    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)

    def resident_boundary(*_args, **_kwargs):
        raise ResidentLoadBoundary

    expert_runtime = _patch_streamed_load_to_live_broker(
        monkeypatch,
        resident_boundary=resident_boundary,
    )
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )

    with pytest.raises(ResidentLoadBoundary):
        load(
            tmp_path,
            mtp=True,
            expert_streaming_config=config,
            expert_manifest=tmp_path / "expert-manifest.json",
            model_config={"model_type": "hy_v3"},
            mtp_artifacts=tmp_path,
        )

    assert expert_runtime.closed is True
