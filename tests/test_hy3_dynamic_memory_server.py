from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_cli import add_expert_streaming_args, expert_streaming_load_kwargs
from mtplx.hy3_q4_context import SingleSequenceGate, admit_hy3_q4_context
from mtplx.server import openai


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
        "--expert-streaming",
        "--expert-memory-limit",
        "110GiB",
        "--expert-max-live-kv-tokens",
        "131072",
        "--expert-cache-scope",
        "global",
        "--expert-slot-layout",
        "component-banks",
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


def test_server_parser_keeps_dynamic_memory_disabled_by_default() -> None:
    args = openai.parse_args(["--warmup-tokens", "0"])

    assert args.hy3_q4_dynamic_memory is False
    assert args.expert_slab_slots is None
    assert args.expert_regrow_hysteresis_slabs is None
    assert args.expert_resize_min_interval_ms is None


def test_server_parser_accepts_explicit_dynamic_memory_tuning() -> None:
    args = openai.parse_args(
        [
            "--warmup-tokens",
            "0",
            "--hy3-q4-dynamic-memory",
            "--expert-slab-slots",
            "64",
            "--expert-regrow-hysteresis-slabs",
            "2",
            "--expert-resize-min-interval-ms",
            "250",
        ]
    )

    assert args.hy3_q4_dynamic_memory is True
    assert args.expert_slab_slots == 64
    assert args.expert_regrow_hysteresis_slabs == 2
    assert args.expert_resize_min_interval_ms == 250


def test_dynamic_memory_opt_in_forces_instrumented_dynamic_expert_config(
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    args = _expert_parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "110GiB",
            "--expert-max-live-kv-tokens",
            "131072",
            "--expert-cache-scope",
            "global",
            "--expert-slot-layout",
            "component-banks",
            "--hy3-q4-dynamic-memory",
            "--expert-slab-slots",
            "64",
            "--expert-regrow-hysteresis-slabs",
            "2",
            "--expert-resize-min-interval-ms",
            "250",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    config = kwargs["expert_streaming_config"]

    assert kwargs["mtp"] is False
    assert config.dynamic_expert_slabs is True
    assert config.resource_telemetry is True
    assert config.expert_slab_slots == 64
    assert config.expert_regrow_hysteresis_slabs == 2
    assert config.expert_resize_min_interval_ms == 250


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


def test_dynamic_memory_tuning_requires_explicit_opt_in_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _hy3_model_root(tmp_path)
    argv = _valid_dynamic_server_argv(root)
    argv.remove("--hy3-q4-dynamic-memory")
    argv.extend(["--expert-slab-slots", "64"])

    _assert_server_rejects_before_load(
        monkeypatch,
        argv,
        match="--expert-slab-slots requires --hy3-q4-dynamic-memory",
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
                "memory_limit_bytes": "110GiB",
                "max_live_kv_tokens": 131072,
                "cache_scope": "global",
                "slot_layout": "component-banks",
                "dynamic_expert_slabs": True,
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
        match="dynamic expert slabs in serving require --hy3-q4-dynamic-memory",
    )


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
            "operating_target_bytes": 110,
            "hard_ceiling_bytes": 112,
            "charged_bytes": 99,
            "logical_expert_records": 128,
            "active_expert_records": 96,
            "resident_expert_records": 80,
            "logical_slab_count": 4,
            "active_slab_count": 3,
            "expert_slab_physical_bytes": 50,
            "pinned_expert_bytes": 7,
            "in_flight_expert_bytes": 5,
            "speculative_expert_bytes": 3,
            "requested_reclaim_bytes": 20,
            "reclaimed_bytes": 18,
            "regrown_bytes": 4,
            "resize_duration_ns": 900,
            "blocked_by_pin_bytes": 2,
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
        "expert_evicted_slabs": 2,
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
    assert payload["operating_target_bytes"] == 110
    assert payload["hard_ceiling_bytes"] == 112
    assert payload["resident_model_bytes"] == 40
    assert payload["kv_representation"] == "q4"
    assert payload["kv_logical_tokens"] == 4096
    assert payload["kv_physical_blocks"] == 64
    assert payload["kv_physical_bytes"] == 11
    assert payload["expert_logical_records"] == 128
    assert payload["expert_active_records"] == 96
    assert payload["expert_resident_records"] == 80
    assert payload["expert_logical_slabs"] == 4
    assert payload["expert_active_slabs"] == 3
    assert payload["expert_physical_bytes"] == 50
    assert payload["inflight_expert_staging_bytes"] == 2
    assert payload["runtime_workspace_bytes"] == 8
    assert payload["evicted_expert_records"] == 9
    assert payload["evicted_expert_slabs"] == 2
    assert payload["admission_failures"] == 1
    assert payload["expert_cache_hit_rate"] == 0.75
    assert payload["ssd_bytes_per_token"] == 1.5
    assert payload["decode_tps"] == 33.5
    assert payload["token_latency_p50_ms"] == 12.0
    assert payload["token_latency_p95_ms"] == 20.0
    assert payload["process_rss_bytes"] == 123
    assert payload["process_compressed_bytes"] is None
    assert payload["system_swap_delta_bytes"] is None
    assert set(payload) == openai.HY3_Q4_DYNAMIC_MEMORY_HEALTH_KEYS
