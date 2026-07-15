from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_cli import add_expert_streaming_args, expert_streaming_load_kwargs
from mtplx.hy3_q4_context import SingleSequenceGate, admit_hy3_q4_context
from mtplx.server import openai


@pytest.fixture(autouse=True)
def _restore_process_environment() -> object:
    """Keep ServerState's intentional process-wide env writes test-local."""

    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def _expert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser, include_hy3_dynamic_memory=True)
    return parser


def _hy3_model_root(tmp_path: Path) -> Path:
    root = tmp_path / "hy3"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "hy_v3"}),
        encoding="utf-8",
    )
    (root / "expert-manifest.json").write_text("{}", encoding="utf-8")
    return root


def _valid_dynamic_server_argv(root: Path) -> list[str]:
    return [
        "--model",
        str(root),
        "--warmup-tokens",
        "0",
        "--generation-mode",
        "ar",
        "--no-load-mtp",
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
        "--hy3-q4-dynamic-context",
        "--paged-kv-quantization",
        "q4",
        "--context-window",
        "131072",
        "--scheduler-mode",
        "serial",
        "--max-active-requests",
        "1",
        "--decode-batch-max",
        "1",
        "--no-session-bank-live-refs",
    ]


def _assert_server_rejects_before_load(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    match: str,
) -> None:
    load_calls: list[object] = []

    def forbidden_load(*args: object, **kwargs: object) -> object:
        load_calls.append((args, kwargs))
        raise AssertionError("load must not run for an invalid dynamic-memory lane")

    monkeypatch.setattr(openai, "load", forbidden_load)
    with pytest.raises(ValueError, match=match):
        openai.ServerState(openai.parse_args(argv))
    assert load_calls == []


def test_dynamic_memory_post_load_rejection_closes_runtime_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ImmediateScheduler:
        def __init__(self, **_kwargs: object) -> None:
            self.shutdown_calls: list[tuple[bool, bool]] = []

        def submit_foreground(
            self,
            function: object,
            *args: object,
            batch_key: str | None = None,
            **kwargs: object,
        ) -> object:
            del batch_key
            from concurrent.futures import Future

            future: Future[object] = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    root = _hy3_model_root(tmp_path)
    runtime = SimpleNamespace(
        model_path=root,
        mtp_enabled=False,
        tokenizer=SimpleNamespace(),
        close_calls=[],
    )

    def close(*, timeout: float | None = None) -> None:
        runtime.close_calls.append(timeout)

    runtime.close = close
    scheduler = ImmediateScheduler()
    monkeypatch.setattr(openai, "ModelWorkScheduler", lambda **_kwargs: scheduler)
    monkeypatch.setattr(openai, "apply_profile_env", lambda *_a, **_k: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda *_a, **_k: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_runtime_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai,
        "_configure_mlx_cache_limit",
        lambda _args: {"configured": False},
    )
    monkeypatch.setattr(openai, "load", lambda *_a, **_k: runtime)
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(
        openai,
        "_resolve_context_window",
        lambda _tokenizer, _model: 32_768,
    )

    with pytest.raises(ValueError, match="loaded model.*131072-token"):
        openai.ServerState(openai.parse_args(_valid_dynamic_server_argv(root)))

    assert runtime.close_calls == [10.0]
    assert scheduler.shutdown_calls[-1] == (False, True)


def test_server_parser_keeps_dynamic_memory_disabled_by_default() -> None:
    args = openai.parse_args(["--warmup-tokens", "0"])

    assert args.hy3_q4_dynamic_memory is False
    assert args.expert_allocator_headroom is None


def test_server_parser_accepts_configurable_memory_limit_and_headroom() -> None:
    args = openai.parse_args(
        [
            "--warmup-tokens",
            "0",
            "--hy3-q4-dynamic-memory",
            "--expert-memory-limit",
            "100GiB",
            "--expert-allocator-headroom",
            "1GiB",
        ]
    )

    assert args.hy3_q4_dynamic_memory is True
    assert args.expert_memory_limit == "100GiB"
    assert args.expert_allocator_headroom == "1GiB"


@pytest.mark.parametrize(
    "removed_flag",
    (
        "--expert-slab-slots",
        "--expert-regrow-hysteresis-slabs",
        "--expert-resize-min-interval-ms",
    ),
)
def test_server_parser_rejects_removed_grouped_allocator_flags(
    removed_flag: str,
) -> None:
    with pytest.raises(SystemExit):
        openai.parse_args(["--warmup-tokens", "0", removed_flag, "1"])


def test_dynamic_memory_opt_in_forces_instrumented_dynamic_expert_config(
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    args = _expert_parser().parse_args(
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

    kwargs = expert_streaming_load_kwargs(args, root)
    config = kwargs["expert_streaming_config"]

    assert kwargs["mtp"] is False
    assert config.dynamic_expert_cache is True
    assert config.resource_telemetry is True
    assert config.kv_bytes_per_token_override == 84_480
    assert config.allocator_headroom_bytes == 1024**3
    assert config.runtime_reserve_bytes == 8 * 1024**3
    assert config.transient_slots == 32
    assert config.cache_policy == "lru"
    assert config.cache_scope == "global"
    assert config.slot_layout == "direct-slots"


def test_dynamic_memory_attestation_rejects_non_q4_physical_kv_geometry() -> None:
    args = SimpleNamespace(
        hy3_q4_dynamic_memory=True,
        hy3_q4_dynamic_context=True,
        expert_streaming=True,
        expert_streaming_config=None,
        expert_manifest=None,
    )
    config = SimpleNamespace(
        dynamic_expert_cache=True,
        resource_telemetry=True,
        kv_bytes_per_token_override=327_680,
    )

    with pytest.raises(ValueError, match="84,480-byte Q4 KV"):
        openai.validate_hy3_q4_dynamic_memory_options(args, config)


def test_dynamic_memory_attestation_requires_one_gib_allocator_headroom() -> None:
    args = SimpleNamespace(
        hy3_q4_dynamic_memory=True,
        hy3_q4_dynamic_context=True,
        expert_streaming=True,
        expert_streaming_config=None,
        expert_manifest=None,
        expert_allocator_headroom=None,
    )
    config = SimpleNamespace(
        dynamic_expert_cache=True,
        resource_telemetry=True,
        kv_bytes_per_token_override=84_480,
        allocator_headroom_bytes=0,
    )

    with pytest.raises(ValueError, match="1 GiB allocator headroom"):
        openai.validate_hy3_q4_dynamic_memory_options(args, config)


def test_dynamic_memory_rejects_non_one_gib_cli_headroom_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)

    _assert_server_rejects_before_load(
        monkeypatch,
        [
            *_valid_dynamic_server_argv(root),
            "--expert-allocator-headroom",
            "2GiB",
        ],
        match="1 GiB allocator headroom",
    )


def test_dynamic_memory_requires_dynamic_context_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    argv = _valid_dynamic_server_argv(root)
    argv.remove("--hy3-q4-dynamic-context")

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="hy3-q4-dynamic-memory requires --hy3-q4-dynamic-context",
    )


def test_dynamic_memory_rejects_mtp_generation_mode_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    argv = _valid_dynamic_server_argv(root)
    argv[argv.index("--generation-mode") + 1] = "mtp"

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="generation-mode ar",
    )


def test_dynamic_memory_rejects_loaded_mtp_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    argv = _valid_dynamic_server_argv(root)
    argv.remove("--no-load-mtp")

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="no-load-mtp",
    )


def test_dynamic_memory_requires_explicit_expert_streaming_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    argv = _valid_dynamic_server_argv(root)
    argv.remove("--expert-streaming")

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="hy3-q4-dynamic-memory requires expert streaming",
    )


def test_dynamic_config_cannot_bypass_serving_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    config_path = tmp_path / "dynamic.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "hy3-q4",
                "memory_limit_bytes": "100GiB",
                "max_live_kv_tokens": 131072,
                "cache_scope": "global",
                "slot_layout": "direct-slots",
                "cache_policy": "lru",
                "dynamic_expert_cache": True,
                "resource_telemetry": True,
            }
        ),
        encoding="utf-8",
    )
    argv = _valid_dynamic_server_argv(root)
    argv.remove("--hy3-q4-dynamic-memory")
    argv.extend(["--expert-streaming-config", str(config_path)])

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="dynamic expert cache in serving requires --hy3-q4-dynamic-memory",
    )


@pytest.mark.parametrize(
    ("profile", "unsafe_key", "unsafe_value", "rejection_term"),
    [
        (
            "performance-cold",
            "MTPLX_VLLM_METAL_PAGED_ATTN",
            "0",
            "paged",
        ),
        (
            "sustained",
            "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW",
            "2048",
            "sliding",
        ),
    ],
    ids=("non-paged-profile", "sliding-window-override"),
)
def test_dynamic_memory_rejects_or_pins_exact_paged_attention_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    unsafe_key: str,
    unsafe_value: str,
    rejection_term: str,
) -> None:
    class ModelLoadBoundary(RuntimeError):
        pass

    paged_key = "MTPLX_VLLM_METAL_PAGED_ATTN"
    sliding_key = "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"
    monkeypatch.delenv(paged_key, raising=False)
    monkeypatch.delenv(sliding_key, raising=False)
    monkeypatch.setenv(unsafe_key, unsafe_value)
    observed: dict[str, str | None] = {}

    def stop_at_model_load(*_args: object, **_kwargs: object) -> object:
        observed[paged_key] = os.environ.get(paged_key)
        observed[sliding_key] = os.environ.get(sliding_key)
        raise ModelLoadBoundary

    monkeypatch.setattr(openai, "load", stop_at_model_load)
    root = _hy3_model_root(tmp_path)
    argv = [*_valid_dynamic_server_argv(root), "--profile", profile]

    try:
        openai.ServerState(openai.parse_args(argv))
    except ValueError as exc:
        assert rejection_term in str(exc).lower()
        return
    except ModelLoadBoundary:
        pass
    else:
        pytest.fail("test model load boundary was not reached")

    if unsafe_key == paged_key:
        assert observed[paged_key] == "1"
    else:
        raw_window = observed[sliding_key]
        assert raw_window is None or not raw_window.strip() or int(raw_window) <= 0


def test_dynamic_memory_diagnostic_ablation_still_pins_exact_q4_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ModelLoadBoundary(RuntimeError):
        pass

    expected = {
        "MTPLX_DYNAMIC_PAGED_KV": "1",
        "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": "1",
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW": "0",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
        "MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS": "1",
        "MTPLX_DYNAMIC_PAGED_KV_MARGIN": "0",
        "MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER": "0",
    }
    unsafe = {
        "MTPLX_DYNAMIC_PAGED_KV": "0",
        "MTPLX_VLLM_METAL_PAGED_ATTN": "0",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "64",
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": "1024",
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW": "2048",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "1",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "fast_sdpa_gather",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "0",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "1",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "64",
        "MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS": "1024",
        "MTPLX_DYNAMIC_PAGED_KV_MARGIN": "128",
        "MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER": "131072",
    }
    for key, value in unsafe.items():
        monkeypatch.setenv(key, value)
    observed: dict[str, str | None] = {}

    def stop_at_model_load(*_args: object, **_kwargs: object) -> object:
        observed.update({key: os.environ.get(key) for key in expected})
        raise ModelLoadBoundary

    monkeypatch.setattr(openai, "load", stop_at_model_load)
    root = _hy3_model_root(tmp_path)
    argv = [
        *_valid_dynamic_server_argv(root),
        "--diagnostic-env-ablation",
    ]

    with pytest.raises(ModelLoadBoundary):
        openai.ServerState(openai.parse_args(argv))

    assert observed == expected


def test_dynamic_memory_disabled_health_shape_is_stable() -> None:
    state = SimpleNamespace(
        args=SimpleNamespace(hy3_q4_dynamic_memory=False),
        runtime=SimpleNamespace(
            expert_resource_telemetry_snapshot=lambda: pytest.fail(
                "disabled health must not sample the runtime"
            )
        ),
        last_metrics=[],
    )

    payload = openai._hy3_q4_dynamic_memory_health(state)

    assert payload["enabled"] is False
    assert set(payload) == openai.HY3_Q4_DYNAMIC_MEMORY_HEALTH_KEYS
    assert all(value is None for key, value in payload.items() if key != "enabled")


def test_dynamic_memory_health_does_not_invent_kv_representation_from_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        args=SimpleNamespace(
            hy3_q4_dynamic_memory=True,
            paged_kv_quantization="q4",
        ),
        runtime=SimpleNamespace(expert_resource_telemetry_snapshot=lambda: {}),
        last_metrics=[],
    )
    monkeypatch.setattr(
        openai,
        "_process_memory_health_snapshot",
        lambda _state: {
            "process_rss_bytes": None,
            "process_compressed_bytes": None,
            "system_swap_delta_bytes": None,
        },
    )

    payload = openai._hy3_q4_dynamic_memory_health(state)

    assert payload["kv_representation"] is None


def test_dynamic_memory_serving_reuses_single_sequence_and_no_live_ref_lane() -> None:
    state = SimpleNamespace(
        args=SimpleNamespace(
            hy3_q4_dynamic_memory=True,
            hy3_q4_dynamic_context=True,
        ),
        hy3_q4_sequence_gate=SingleSequenceGate(),
    )
    admission = admit_hy3_q4_context(
        rendered_input_tokens=4096,
        requested_output_tokens=128,
    )

    assert openai._hy3_q4_session_bank_policy(
        state,
        session_bank=object(),
        session_keep_live_ref=True,
    ) == (None, False)
    lease = openai._acquire_hy3_q4_sequence(
        state,
        request_id="request-1",
        admission=admission,
    )
    with pytest.raises(openai.HTTPException) as exc_info:
        openai._acquire_hy3_q4_sequence(
            state,
            request_id="request-2",
            admission=admission,
        )
    assert exc_info.value.status_code == 429
    assert lease is not None
    assert lease.release("complete") is True


def test_dynamic_memory_health_maps_runtime_data_without_inventing_os_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_snapshot = {
        "dynamic_memory": {
            "memory_limit_bytes": 100,
            "allocator_headroom_bytes": 1,
            "classified_limit_bytes": 99,
            "classified_bytes": 91,
            "charged_bytes": 99,
            "charged_residual_bytes": 1,
            "logical_expert_records": 128,
            "allocated_record_count": 96,
            "active_expert_records": 96,
            "resident_expert_records": 80,
            "expert_cache_physical_bytes": 50,
            "pinned_expert_bytes": 7,
            "in_flight_expert_bytes": 5,
            "speculative_expert_bytes": 3,
            "record_allocations": 11,
            "record_reuses": 5,
            "record_evictions": 9,
            "record_releases": 3,
            "allocator_active_bytes": 88,
            "allocator_cache_bytes": 6,
            "allocator_peak_bytes": 101,
        },
        "memory_broker": {
            "resident_model_bytes": 40,
            "kv_physical_bytes": 11,
            "in_flight_expert_staging_bytes": 2,
            "runtime_workspace_bytes": 8,
            "admission_failure_count": 1,
        },
        "cache": {
            "hit_rate": 0.75,
            "evictions": 9,
            "bytes_read": 4096,
        },
        "kv": {
            "representation": "q4",
            "logical_tokens": 4096,
            "physical_blocks": 64,
            "physical_bytes": 11,
        },
        "python_control_cpu": {"clock": "thread_time_ns"},
        "ssd_bytes_per_token": 1.5,
        "token_latency_p50_ms": 12.0,
        "token_latency_p95_ms": 20.0,
    }
    state = SimpleNamespace(
        args=SimpleNamespace(
            hy3_q4_dynamic_memory=True,
            paged_kv_quantization="q4",
        ),
        runtime=SimpleNamespace(
            expert_resource_telemetry_snapshot=lambda: resource_snapshot
        ),
        last_metrics=[{"decode_tok_s": 33.5}],
    )
    monkeypatch.setattr(
        openai,
        "_process_memory_health_snapshot",
        lambda _state: {
            "process_rss_bytes": 123,
            "process_compressed_bytes": None,
            "system_swap_delta_bytes": None,
        },
    )

    payload = openai._hy3_q4_dynamic_memory_health(state)

    assert payload["enabled"] is True
    assert payload["memory_limit_bytes"] == 100
    assert payload["allocator_headroom_bytes"] == 1
    assert payload["classified_limit_bytes"] == 99
    assert payload["classified_bytes"] == 91
    assert payload["resident_model_bytes"] == 40
    assert payload["charged_residual_bytes"] == 1
    assert payload["kv_representation"] == "q4"
    assert payload["kv_logical_tokens"] == 4096
    assert payload["kv_physical_blocks"] == 64
    assert payload["kv_physical_bytes"] == 11
    assert payload["expert_logical_records"] == 128
    assert payload["expert_allocated_records"] == 96
    assert payload["expert_active_records"] == 96
    assert payload["expert_resident_records"] == 80
    assert payload["expert_physical_bytes"] == 50
    assert payload["inflight_expert_staging_bytes"] == 2
    assert payload["runtime_workspace_bytes"] == 8
    assert payload["record_allocations"] == 11
    assert payload["record_reuses"] == 5
    assert payload["record_evictions"] == 9
    assert payload["record_releases"] == 3
    assert payload["admission_failures"] == 1
    assert payload["expert_cache_hit_rate"] == 0.75
    assert payload["ssd_bytes_per_token"] == 1.5
    assert payload["decode_tps"] == 33.5
    assert payload["token_latency_p50_ms"] == 12.0
    assert payload["token_latency_p95_ms"] == 20.0
    assert payload["python_control_cpu"] == {"clock": "thread_time_ns"}
    assert payload["process_rss_bytes"] == 123
    assert payload["process_compressed_bytes"] is None
    assert payload["system_swap_delta_bytes"] is None
    assert set(payload) == openai.HY3_Q4_DYNAMIC_MEMORY_HEALTH_KEYS


def test_dynamic_memory_health_keeps_startup_logical_ownership_separate_from_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_snapshot = {
        "kv": {
            "representation": "q4",
            "logical_tokens": 1,
            "physical_blocks": 1,
            "physical_bytes": 1_351_680,
        }
    }
    state = SimpleNamespace(
        args=SimpleNamespace(
            hy3_q4_dynamic_memory=True,
            paged_kv_quantization="q4",
        ),
        runtime=SimpleNamespace(
            expert_resource_telemetry_snapshot=lambda: resource_snapshot
        ),
        last_metrics=[],
    )
    monkeypatch.setattr(
        openai,
        "_process_memory_health_snapshot",
        lambda _state: {
            "process_rss_bytes": None,
            "process_compressed_bytes": None,
            "system_swap_delta_bytes": None,
        },
    )

    payload = openai._hy3_q4_dynamic_memory_health(state)

    assert payload["kv_logical_tokens"] == 1
    assert payload["kv_physical_blocks"] == 1
    assert payload["kv_physical_bytes"] == 1_351_680
