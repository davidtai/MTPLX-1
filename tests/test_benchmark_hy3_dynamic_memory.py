from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    AllocatorSample,
    BenchmarkGateError,
    CacheStartState,
    ProbeSlab,
    QwenIsolationHooks,
    balanced_campaign_schedule,
    canonical_sha256,
    run_allocator_release_probe,
    run_balanced_campaign,
    run_exclusive_hardware_window,
    validate_allocator_probe,
    validate_campaign_observation,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity() -> dict[str, object]:
    return {
        "model_key": "hy3-q4",
        "model_artifact_sha256": "a" * 64,
        "expert_manifest_sha256": "b" * 64,
        "source_git_commit": "c" * 40,
        "arm_config_sha256": "d" * 64,
        "normalized_config_sha256": "e" * 64,
        "kv_quantization": "q4",
        "kv_block_size_tokens": 16,
        "total_context_tokens": 131_072,
    }


def _slot_health() -> dict[str, int]:
    return {
        "active_routes": 0,
        "pins": 0,
        "loading": 0,
        "failed": 0,
        "integrity_errors": 0,
        "completion_fence_failures": 0,
    }


def _point(
    phase: str,
    timestamp_ns: int,
    *,
    expert: int,
    kv: int,
    active: int | None = None,
    cache: int = 0,
) -> dict[str, object]:
    health = _slot_health()
    return {
        "phase": phase,
        "monotonic_ns": timestamp_ns,
        "allocator_active_bytes": expert + kv if active is None else active,
        "allocator_cache_bytes": cache,
        "allocator_peak_bytes": expert + kv + cache,
        "expert_slab_physical_bytes": expert,
        "kv_physical_bytes": kv,
        "slot_health": health,
        "slot_health_sha256": _sha(health),
    }


def _observation(
    arm: str,
    context_tokens: int,
    repetition: int,
    *,
    tok_s: float,
) -> dict[str, object]:
    tokens = [101, context_tokens, 202]
    routes = [[0, 3, 7], [1, 2, 9]]
    if arm == "dynamic":
        timeline = [
            _point("pre_growth", 1, expert=800, kv=100),
            _point("post_expert_reclaim", 2, expert=600, kv=100),
            _point("post_kv_growth", 3, expert=600, kv=300),
            _point("hold", 4, expert=600, kv=300),
            _point("hold", 5, expert=600, kv=300),
            _point("hold", 6, expert=600, kv=300),
            _point("post_reset", 7, expert=600, kv=0),
            _point("post_regrow", 8, expert=800, kv=0),
        ]
        start = {"kind": "empty-q4", "kv_physical_bytes": 100, "kv_blocks": 1}
    else:
        timeline = [
            _point("pre_growth", 1, expert=600, kv=300),
            _point("post_kv_growth", 2, expert=600, kv=300),
            _point("hold", 3, expert=600, kv=300),
            _point("hold", 4, expert=600, kv=300),
            _point("hold", 5, expert=600, kv=300),
            _point("post_reset", 6, expert=600, kv=0),
        ]
        start = {
            "kind": "static-128k-reserved-q4",
            "kv_physical_bytes": 300,
            "kv_blocks": 8192,
        }
    identity = _identity()
    identity["arm_config_sha256"] = ("d" if arm == "static" else "f") * 64
    return {
        "schema": "mtplx-hy3-dynamic-memory-observation-v1",
        "arm": arm,
        "context_tokens": context_tokens,
        "repetition": repetition,
        "cache_start_state": start,
        "identity": identity,
        "prompt_sha256": "1" * 64,
        "generated_token_ids": tokens,
        "generated_token_sha256": _sha(tokens),
        "route_trace": routes,
        "route_trace_sha256": _sha(routes),
        "expert_hashes": {"0:3": "2" * 64, "1:2": "3" * 64},
        "timeline": timeline,
        "metrics": {
            "generated_tokens": len(tokens),
            "elapsed_seconds": len(tokens) / tok_s,
            "tokens_per_second": tok_s,
            "peak_charged_bytes": 900,
        },
    }


def _probe_result(*, released: int = 128, untouched_executable: bool = True):
    return {
        "schema": "mtplx-hy3-allocator-release-probe-v1",
        "manifest": {
            "identity": _identity(),
            "selected_slab_id": "slab-a",
            "selected_registered_physical_bytes": 128,
            "untouched_slab_id": "slab-b",
            "allocated_slab_count": 2,
        },
        "before_release": {
            "active_bytes": 1_000,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "after_release": {
            "active_bytes": 1_000 - released,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "untouched_slab_executable": untouched_executable,
        "error": None,
    }


def test_context_matrix_and_schedule_balance_both_physical_orders() -> None:
    assert CONTEXT_MATRIX_TOKENS == (4096, 32768, 65536, 131072)

    schedule = balanced_campaign_schedule(repetitions=4)

    assert len(schedule) == 4 * 4 * 2
    for context in CONTEXT_MATRIX_TOKENS:
        rows = [row for row in schedule if row.context_tokens == context]
        assert [row.arms for row in rows[::2]] == [
            ("static", "dynamic"),
            ("dynamic", "static"),
            ("static", "dynamic"),
            ("dynamic", "static"),
        ]
        assert sum(row.arm == "static" and row.order_index == 0 for row in rows) == 2
        assert sum(row.arm == "dynamic" and row.order_index == 0 for row in rows) == 2


def test_schedule_rejects_an_unbalanced_repetition_count() -> None:
    with pytest.raises(ValueError, match="positive even"):
        balanced_campaign_schedule(repetitions=3)


def test_observation_requires_declared_start_and_physical_ordering() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)

    validated = validate_campaign_observation(row)

    assert isinstance(validated.cache_start_state, CacheStartState)
    assert validated.context_tokens == 4096

    reordered = _observation("dynamic", 4096, 0, tok_s=12.0)
    reordered["timeline"][1]["kv_physical_bytes"] = 200
    with pytest.raises(BenchmarkGateError, match="before KV"):
        validate_campaign_observation(reordered)


def test_observation_requires_stable_hold_reset_regrow_and_block_crossing() -> None:
    unstable = _observation("dynamic", 4096, 0, tok_s=12.0)
    unstable["timeline"][4]["allocator_cache_bytes"] = 1
    with pytest.raises(BenchmarkGateError, match="stable hold"):
        validate_campaign_observation(unstable)

    no_reset = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_reset["timeline"] = no_reset["timeline"][:-2]
    with pytest.raises(BenchmarkGateError, match="post_reset"):
        validate_campaign_observation(no_reset)

    no_regrow = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_regrow["timeline"] = no_regrow["timeline"][:-1]
    with pytest.raises(BenchmarkGateError, match="post_regrow"):
        validate_campaign_observation(no_regrow)

    no_boundary = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_boundary["cache_start_state"]["kv_blocks"] = 255
    no_boundary["timeline"][2]["kv_physical_bytes"] = 110
    with pytest.raises(BenchmarkGateError, match="block boundaries"):
        validate_campaign_observation(no_boundary)


def test_observation_recomputes_token_route_and_slot_health_hashes() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["generated_token_ids"].append(303)
    with pytest.raises(BenchmarkGateError, match="generated_token_sha256"):
        validate_campaign_observation(row)

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["route_trace"][0].append(11)
    with pytest.raises(BenchmarkGateError, match="route_trace_sha256"):
        validate_campaign_observation(row)

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["timeline"][-1]["slot_health"]["failed"] = 1
    with pytest.raises(BenchmarkGateError, match="slot health"):
        validate_campaign_observation(row)


def test_allocator_probe_requires_real_charged_release_and_untouched_execution() -> (
    None
):
    passed = validate_allocator_probe(_probe_result())
    assert passed["gate_passed"] is True
    assert passed["charged_release_bytes"] == 128

    with pytest.raises(BenchmarkGateError, match="charged allocator"):
        validate_allocator_probe(_probe_result(released=127))
    with pytest.raises(BenchmarkGateError, match="untouched slab"):
        validate_allocator_probe(_probe_result(untouched_executable=False))


def test_allocator_probe_samples_two_evaluated_slabs_and_releases_one() -> None:
    calls: list[object] = []
    samples = iter(
        (
            AllocatorSample(active_bytes=100, cache_bytes=20, peak_bytes=120),
            AllocatorSample(active_bytes=300, cache_bytes=20, peak_bytes=320),
            AllocatorSample(active_bytes=200, cache_bytes=20, peak_bytes=320),
        )
    )
    slabs = (
        ProbeSlab("slab-a", registered_physical_bytes=100),
        ProbeSlab("slab-b", registered_physical_bytes=100),
    )

    result = run_allocator_release_probe(
        identity=_identity(),
        allocate_slabs=lambda: calls.append("allocate") or slabs,
        evaluate_slabs=lambda actual: calls.append(("evaluate", actual)),
        sample_allocator=lambda: next(samples),
        release_slab=lambda slab: calls.append(("release", slab.slab_id)),
        execute_slab=lambda slab: calls.append(("execute", slab.slab_id)) or True,
    )

    assert result["gate_passed"] is True
    assert calls == [
        "allocate",
        ("evaluate", slabs),
        ("release", "slab-a"),
        ("execute", "slab-b"),
    ]


def test_allocator_probe_fails_closed_on_injected_errors() -> None:
    result = run_allocator_release_probe(
        identity=_identity(),
        allocate_slabs=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        evaluate_slabs=lambda _slabs: None,
        sample_allocator=lambda: AllocatorSample(0, 0, 0),
        release_slab=lambda _slab: None,
        execute_slab=lambda _slab: True,
    )

    assert result["gate_passed"] is False
    assert "injected" in result["error"]


def test_balanced_campaign_retains_raw_pairs_and_confidence_intervals() -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int) -> Mapping[str, object]:
        base = 10.0 + repetition
        return _observation(
            arm,
            context_tokens,
            repetition,
            tok_s=base * (1.1 if arm == "dynamic" else 1.0),
        )

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=4,
        bootstrap_resamples=500,
        bootstrap_seed=46,
    )

    assert result["schema"] == "mtplx-hy3-dynamic-memory-campaign-v1"
    assert len(result["raw_observations"]) == 32
    assert len(result["paired_samples"]) == 16
    assert set(result["contexts"]) == set(CONTEXT_MATRIX_TOKENS)
    for summary in result["paired_metrics_by_context"].values():
        assert summary["sample_count"] == 4
        assert summary["dynamic_vs_static_tps_ratio"]["mean"] == pytest.approx(1.1)
        low, high = summary["dynamic_vs_static_tps_ratio"]["confidence_interval_95"]
        assert low <= 1.1 <= high


def test_campaign_fails_before_arm_execution_when_probe_gate_did_not_pass() -> None:
    calls: list[object] = []
    with pytest.raises(BenchmarkGateError, match="allocator-release probe"):
        run_balanced_campaign(
            allocator_probe={"gate_passed": False},
            execute_arm=lambda *args: calls.append(args),
            repetitions=2,
        )
    assert calls == []


def test_campaign_rejects_pair_identity_or_output_drift() -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int):
        row = _observation(arm, context_tokens, repetition, tok_s=10.0)
        if arm == "dynamic" and context_tokens == 4096 and repetition == 0:
            row["generated_token_ids"] = [999]
            row["generated_token_sha256"] = canonical_sha256([999])
            row["metrics"]["generated_tokens"] = 1
            row["metrics"]["elapsed_seconds"] = 0.1
        return row

    with pytest.raises(BenchmarkGateError, match="paired generated tokens"):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
        )


def test_qwen_isolation_restores_and_releases_lane_on_workload_exception() -> None:
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: calls.append(("unload", state)),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    with pytest.raises(RuntimeError, match="workload failed"):
        run_exclusive_hardware_window(
            lambda: (_ for _ in ()).throw(RuntimeError("workload failed")),
            hooks=hooks,
        )

    assert calls == [
        "acquire",
        "capture",
        ("unload", captured),
        ("restore", captured),
        ("verify", captured),
        "release",
    ]


def test_qwen_isolation_restores_even_when_unload_raises() -> None:
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: (
            calls.append(("unload", state))
            or (_ for _ in ()).throw(RuntimeError("unload failed"))
        ),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    with pytest.raises(RuntimeError, match="unload failed"):
        run_exclusive_hardware_window(lambda: calls.append("workload"), hooks=hooks)

    assert "workload" not in calls
    assert calls[-3:] == [("restore", captured), ("verify", captured), "release"]


def test_cli_plan_declares_exact_matrix_balanced_order_and_qwen_hooks(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "probe_command": ["probe", "--json"],
                "arm_command_template": [
                    "arm",
                    "--arm",
                    "{arm}",
                    "--context",
                    "{context_tokens}",
                    "--repetition",
                    "{repetition}",
                ],
                "repetitions": 2,
                "qwen": {
                    "acquire_lane_command": ["lane", "acquire"],
                    "release_lane_command": ["lane", "release"],
                    "capture_command": ["qwen", "capture"],
                    "unload_command": ["qwen", "unload"],
                    "restore_command": ["qwen", "restore"],
                    "verify_command": ["qwen", "verify"],
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--spec", str(spec_path), "--plan-only"],
        cwd=script.parent.parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["context_matrix_tokens"] == list(CONTEXT_MATRIX_TOKENS)
    assert plan["qwen_isolation_configured"] is True
    assert [row["arm"] for row in plan["schedule"][:4]] == [
        "static",
        "dynamic",
        "dynamic",
        "static",
    ]
