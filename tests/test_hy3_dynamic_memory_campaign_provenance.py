from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from mtplx.benchmarks.runners.hy3_dynamic_memory import BenchmarkGateError


PROBE_SOURCE = "benchmarks/probe_hy3_component_slabs.py"
ARM_SOURCE = "benchmarks/observe_hy3_dynamic_memory_arm.py"
QUALITY_SOURCE = "benchmarks/benchmark_hy3_kv_quality.py"
QWEN_SOURCE = "benchmarks/hy3_qwen_isolation.py"
SPEC_SOURCE = "benchmarks/specs/issue46.json"
HOOKS_CONFIG_SOURCE = "benchmarks/specs/issue46-hooks.json"
_CAMPAIGN_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "benchmark_hy3_dynamic_memory.py"
)


def _load_campaign_module():
    spec = importlib.util.spec_from_file_location(
        "issue46_campaign_provenance_target",
        _CAMPAIGN_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign_module = _load_campaign_module()


def test_run_spec_reuses_one_explicit_owner_for_every_qwen_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen = {
        "acquire_lane_command": ("acquire",),
        "release_lane_command": ("release",),
        "capture_command": ("capture",),
        "unload_command": ("unload",),
        "restore_command": ("restore",),
        "verify_command": ("verify",),
    }
    spec = {
        "legacy_exclusive_lane_lock": str(tmp_path / "legacy-gpu.lock"),
        "qwen": qwen,
        "artifact_verify_command": ("verify-artifact",),
        "probe_command": ("probe",),
        "quality_command": ("quality",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
        "workload_subprocess_timeout_seconds": 1234,
        "qwen_control_timeout_seconds": 67,
        "subprocess_termination_grace_seconds": 5,
    }
    calls: list[tuple[str, dict[str, str] | None, object, object]] = []
    real_lane_parent = tmp_path / "real-lanes"
    real_lane_parent.mkdir()
    lane_alias = tmp_path / "lane-alias"
    lane_alias.symlink_to(real_lane_parent, target_is_directory=True)
    lane = real_lane_parent / "exclusive-lane"
    reported_lane = lane_alias / lane.name
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    captured = {"loaded": False, "models": []}
    journal_payloads: list[dict[str, object]] = []
    workload_calls: list[dict[str, object]] = []

    def fake_json_subprocess(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> dict[str, object]:
        action = command[0]
        env = kwargs.get("env")
        assert env is None or isinstance(env, dict)
        calls.append(
            (
                action,
                env,
                kwargs.get("timeout_seconds"),
                kwargs.get("termination_grace_seconds"),
            )
        )
        if action == "acquire":
            lane.mkdir()
            return {
                "acquired": True,
                "lane": str(reported_lane),
                "owner": owner,
            }
        if action == "capture":
            return captured
        if action == "unload":
            journal = lane / "issue46-recovery.json"
            journal_payloads.append(json.loads(journal.read_text(encoding="utf-8")))
            return {"loaded": False, "models": [], "unloaded": True}
        if action == "restore":
            return {**captured, "restored": True}
        if action == "verify":
            return {**captured, "restored": True}
        if action == "release":
            assert not (lane / "issue46-recovery.json").exists()
            lane.rmdir()
            return {"lane": str(reported_lane), "released": True, "owner": owner}
        raise AssertionError(action)

    monkeypatch.setattr(campaign_module.secrets, "token_hex", lambda _size: "b" * 64)
    monkeypatch.setattr(campaign_module, "run_json_subprocess", fake_json_subprocess)
    monkeypatch.setattr(
        campaign_module,
        "run_subprocess_campaign",
        lambda **kwargs: workload_calls.append(kwargs) or {"status": "passed"},
    )

    result = campaign_module.run_spec(spec, cwd=tmp_path)

    assert result["status"] == "passed"
    evidence = result["qwen_isolation"]
    assert evidence["schema"] == "mtplx-issue46-qwen-isolation-v1"
    assert evidence["acquire"] == {
        "acquired": True,
        "lane": str(lane),
        "owner": owner,
    }
    assert evidence["capture"] == captured
    assert evidence["unload"] == {
        "loaded": False,
        "models": [],
        "unloaded": True,
    }
    assert evidence["restore"] == {**captured, "restored": True}
    assert evidence["verify"] == {**captured, "restored": True}
    assert evidence["release"] == {
        "lane": str(lane),
        "released": True,
        "owner": owner,
    }
    assert evidence["legacy_exclusive_lane_lock"] == {
        "path": str(tmp_path / "legacy-gpu.lock"),
        "acquired": True,
        "released": True,
    }
    journal_evidence = evidence["recovery_journal"]
    assert journal_evidence["path"] == str(lane / "issue46-recovery.json")
    assert journal_evidence["durable_before_unload"] is True
    assert journal_evidence["removed_after_restore_verification"] is True
    assert len(journal_evidence["sha256"]) == 64
    assert journal_payloads == [
        {
            "captured_state": captured,
            "owner": owner,
            "schema": "mtplx-issue46-qwen-recovery-v1",
        }
    ]
    assert not lane.exists()

    qwen_calls = calls
    assert [action for action, _env, _timeout, _grace in qwen_calls] == [
        "acquire",
        "capture",
        "unload",
        "restore",
        "verify",
        "release",
    ]
    expected_env = {
        "MTPLX_ISSUE46_CAMPAIGN_OWNER_TOKEN": "b" * 64,
        "MTPLX_ISSUE46_CAMPAIGN_PID": str(os.getpid()),
    }
    assert all(env == expected_env for _action, env, _timeout, _grace in qwen_calls)
    assert all(timeout == 67 for _action, _env, timeout, _grace in qwen_calls)
    assert all(grace == 5 for _action, _env, _timeout, grace in qwen_calls)
    assert len(workload_calls) == 1
    assert workload_calls[0]["quality_command"] == ("quality",)
    assert workload_calls[0]["subprocess_timeout_seconds"] == 1234
    assert workload_calls[0]["subprocess_termination_grace_seconds"] == 5


@pytest.mark.parametrize("value", (None, "", 46))
def test_release_lane_rejects_a_missing_or_non_string_path(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(BenchmarkGateError, match="release result omitted its lane"):
        campaign_module._canonical_release_lane(
            value,
            acquired_lane=tmp_path / "acquired-lane",
        )


def test_release_lane_rejects_a_different_canonical_path(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkGateError, match="differs from acquisition"):
        campaign_module._canonical_release_lane(
            str(tmp_path / "released-lane"),
            acquired_lane=(tmp_path / "acquired-lane").resolve(),
        )


def test_release_lane_rejects_release_before_acquisition(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkGateError, match="before lane acquisition"):
        campaign_module._canonical_release_lane(
            str(tmp_path / "released-lane"),
            acquired_lane=None,
        )


def test_run_spec_refuses_to_overlap_a_held_legacy_gpu_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "legacy_exclusive_lane_lock": str(tmp_path / "legacy-gpu.lock"),
        "qwen": {
            "acquire_lane_command": ("acquire",),
            "release_lane_command": ("release",),
            "capture_command": ("capture",),
            "unload_command": ("unload",),
            "restore_command": ("restore",),
            "verify_command": ("verify",),
        },
        "artifact_verify_command": ("verify-artifact",),
        "probe_command": ("probe",),
        "quality_command": ("quality",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
    }

    def reject_lock(_descriptor: int, operation: int) -> None:
        assert operation & campaign_module.fcntl.LOCK_NB
        raise BlockingIOError("legacy benchmark owns the GPU")

    monkeypatch.setattr(campaign_module.fcntl, "flock", reject_lock)
    monkeypatch.setattr(
        campaign_module,
        "run_json_subprocess",
        lambda *_args, **_kwargs: pytest.fail(
            "Qwen transitions must not start while the legacy lane is held"
        ),
    )

    with pytest.raises(BenchmarkGateError, match="legacy.*GPU lane is active"):
        campaign_module.run_spec(spec, cwd=tmp_path)


def test_legacy_gpu_lane_wait_retries_before_qwen_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[int] = []
    sleeps: list[float] = []

    def flock(_descriptor: int, operation: int) -> None:
        operations.append(operation)
        if len(operations) == 1:
            raise BlockingIOError("legacy lane is busy")

    monotonic_values = iter((10.0, 10.0, 10.25))
    monkeypatch.setattr(campaign_module.fcntl, "flock", flock)
    monkeypatch.setattr(
        campaign_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)

    descriptor = campaign_module._acquire_legacy_lane_lock(
        tmp_path / "legacy-gpu.lock",
        wait_seconds=1.0,
    )
    os.close(descriptor)

    assert operations == [
        campaign_module.fcntl.LOCK_EX | campaign_module.fcntl.LOCK_NB,
        campaign_module.fcntl.LOCK_EX | campaign_module.fcntl.LOCK_NB,
    ]
    assert sleeps == [0.25]


def test_legacy_gpu_lane_wait_closes_descriptor_when_deadline_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = campaign_module.os.open
    real_close = campaign_module.os.close
    opened: list[int] = []
    closed: list[int] = []

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(campaign_module.os, "open", tracked_open)
    monkeypatch.setattr(campaign_module.os, "close", tracked_close)
    monkeypatch.setattr(
        campaign_module.time,
        "monotonic",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        campaign_module._acquire_legacy_lane_lock(
            tmp_path / "legacy-gpu.lock",
            wait_seconds=1.0,
        )

    assert opened == closed


def test_legacy_gpu_lane_release_closes_descriptor_when_unlock_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(
        campaign_module.fcntl,
        "flock",
        lambda _descriptor, _operation: (_ for _ in ()).throw(OSError("unlock")),
    )
    monkeypatch.setattr(campaign_module.os, "close", closed.append)

    with pytest.raises(OSError, match="unlock"):
        campaign_module._release_legacy_lane_lock(2468)

    assert closed == [2468]


def test_run_spec_retains_durable_qwen_recovery_journal_when_verify_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    captured = {"loaded": True, "models": ["qwen"]}
    spec = {
        "qwen": {
            "acquire_lane_command": ("acquire",),
            "release_lane_command": ("release",),
            "capture_command": ("capture",),
            "unload_command": ("unload",),
            "restore_command": ("restore",),
            "verify_command": ("verify",),
        },
        "artifact_verify_command": ("verify-artifact",),
        "probe_command": ("probe",),
        "quality_command": ("quality",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
    }

    def fake_json_subprocess(command: tuple[str, ...], **_kwargs: object):
        action = command[0]
        if action == "acquire":
            lane.mkdir()
            return {"acquired": True, "lane": str(lane), "owner": owner}
        if action == "capture":
            return captured
        if action == "unload":
            return {"loaded": False, "models": [], "unloaded": True}
        if action == "restore":
            return {**captured, "restored": True}
        if action == "verify":
            return {**captured, "restored": False}
        if action == "release":
            pytest.fail("lane must remain owned when restoration is not proven")
        raise AssertionError(action)

    monkeypatch.setattr(campaign_module, "run_json_subprocess", fake_json_subprocess)
    monkeypatch.setattr(
        campaign_module,
        "run_subprocess_campaign",
        lambda **_kwargs: {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match="Qwen verify"):
        campaign_module.run_spec(spec, cwd=tmp_path)

    journal = lane / "issue46-recovery.json"
    assert json.loads(journal.read_text(encoding="utf-8")) == {
        "captured_state": captured,
        "owner": owner,
        "schema": "mtplx-issue46-qwen-recovery-v1",
    }


def test_run_spec_never_unloads_qwen_when_recovery_journal_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    captured = {"loaded": True, "models": ["qwen"]}
    calls: list[str] = []
    spec = {
        "qwen": {
            "acquire_lane_command": ("acquire",),
            "release_lane_command": ("release",),
            "capture_command": ("capture",),
            "unload_command": ("unload",),
            "restore_command": ("restore",),
            "verify_command": ("verify",),
        },
        "artifact_verify_command": ("verify-artifact",),
        "probe_command": ("probe",),
        "quality_command": ("quality",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
    }

    def fake_json_subprocess(command: tuple[str, ...], **_kwargs: object):
        action = command[0]
        calls.append(action)
        if action == "acquire":
            lane.mkdir()
            return {"acquired": True, "lane": str(lane), "owner": owner}
        if action == "capture":
            return captured
        if action == "release":
            assert not any(lane.iterdir())
            lane.rmdir()
            return {"lane": str(lane), "released": True, "owner": owner}
        pytest.fail(f"unexpected Qwen transition after journal failure: {action}")

    monkeypatch.setattr(campaign_module, "run_json_subprocess", fake_json_subprocess)
    monkeypatch.setattr(
        campaign_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(BenchmarkGateError, match="recovery journal.*fsync failed"):
        campaign_module.run_spec(spec, cwd=tmp_path)

    assert calls == ["acquire", "capture", "release"]
    assert not lane.exists()


def test_recovery_journal_cleanup_preserves_termination_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "exclusive-lane"
    lane.mkdir()
    monkeypatch.setattr(
        campaign_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        campaign_module._write_qwen_recovery_journal(
            lane=lane,
            owner={"campaign_pid": 1234},
            captured_state={"loaded": False, "models": []},
        )

    assert not (lane / "issue46-recovery.json").exists()


def test_run_spec_restores_qwen_after_post_campaign_attestation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "qwen": {
            "acquire_lane_command": ("acquire",),
            "release_lane_command": ("release",),
            "capture_command": ("capture",),
            "unload_command": ("unload",),
            "restore_command": ("restore",),
            "verify_command": ("verify",),
        },
        "artifact_verify_command": ("verify-artifact",),
        "probe_command": ("probe",),
        "quality_command": ("quality",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
    }
    calls: list[str] = []
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    captured = {"loaded": True, "models": ["qwen"]}

    def fake_json_subprocess(command: tuple[str, ...], **_kwargs: object):
        action = command[0]
        calls.append(action)
        if action == "acquire":
            lane.mkdir()
            return {"acquired": True, "lane": str(lane), "owner": owner}
        if action == "capture":
            return captured
        if action == "unload":
            return {"loaded": False, "models": [], "unloaded": True}
        if action in {"restore", "verify"}:
            return {**captured, "restored": True}
        if action == "release":
            lane.rmdir()
            return {"lane": str(lane), "released": True, "owner": owner}
        raise AssertionError(action)

    monkeypatch.setattr(campaign_module, "run_json_subprocess", fake_json_subprocess)
    monkeypatch.setattr(
        campaign_module,
        "run_subprocess_campaign",
        lambda **_kwargs: (_ for _ in ()).throw(
            BenchmarkGateError("post-campaign artifact drift")
        ),
    )

    with pytest.raises(BenchmarkGateError, match="post-campaign artifact drift"):
        campaign_module.run_spec(spec, cwd=tmp_path)

    assert calls == [
        "acquire",
        "capture",
        "unload",
        "restore",
        "verify",
        "release",
    ]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _default_spec() -> dict[str, object]:
    qwen = {
        f"{action}_command": [
            "uv",
            "run",
            "python",
            QWEN_SOURCE,
            action.removesuffix("_lane"),
        ]
        for action in (
            "acquire_lane",
            "release_lane",
            "capture",
            "unload",
            "restore",
            "verify",
        )
    }
    return {
        "legacy_exclusive_lane_lock": "/tmp/mtplx-gpu-exclusive.lock",
        "legacy_exclusive_lane_wait_seconds": 0,
        "artifact_verify_command": [
            "uv",
            "run",
            "python",
            PROBE_SOURCE,
            "--hooks-config",
            HOOKS_CONFIG_SOURCE,
            "--verify-artifact-only",
        ],
        "probe_command": [
            "uv",
            "run",
            "python",
            PROBE_SOURCE,
            "--hooks-config",
            HOOKS_CONFIG_SOURCE,
            "--json",
        ],
        "quality_command": [
            "uv",
            "run",
            "python",
            QUALITY_SOURCE,
            "--hooks-config",
            HOOKS_CONFIG_SOURCE,
        ],
        "arm_command_template": [
            "uv",
            "run",
            "python",
            ARM_SOURCE,
            "--hooks-config",
            HOOKS_CONFIG_SOURCE,
            "--arm",
            "{arm}",
            "--context-tokens",
            "{context_tokens}",
            "--repetition",
            "{repetition}",
        ],
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
        "qwen": qwen,
    }


def _create_campaign_repo(
    tmp_path: Path,
    *,
    mutate_spec: Callable[[dict[str, object]], None] | None = None,
    track_spec: bool = True,
    track_hooks_config: bool = True,
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in (PROBE_SOURCE, ARM_SOURCE, QUALITY_SOURCE, QWEN_SOURCE):
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# tracked source: {relative}\n", encoding="utf-8")

    spec = _default_spec()
    if mutate_spec is not None:
        mutate_spec(spec)
    spec_path = repo / SPEC_SOURCE
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hooks_config = repo / HOOKS_CONFIG_SOURCE
    hooks_config.parent.mkdir(parents=True, exist_ok=True)
    hooks_config.write_text('{"frozen":"issue46"}\n', encoding="utf-8")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Issue 46 Test")
    _git(repo, "config", "user.email", "issue46@example.invalid")
    tracked = [PROBE_SOURCE, ARM_SOURCE, QUALITY_SOURCE, QWEN_SOURCE]
    if track_hooks_config:
        tracked.append(HOOKS_CONFIG_SOURCE)
    if track_spec:
        tracked.append(SPEC_SOURCE)
    _git(repo, "add", "--", *tracked)
    _git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "frozen issue46 campaign",
    )
    return repo, spec_path


def test_campaign_spec_rejects_a_decoy_legacy_gpu_lock() -> None:
    spec = _default_spec()
    spec["legacy_exclusive_lane_lock"] = "/tmp/not-the-shared-gpu-lane.lock"

    with pytest.raises(BenchmarkGateError, match="shared MTPLX GPU lock"):
        campaign_module._parse_spec_bytes(
            json.dumps(spec).encode("utf-8"),
            path=Path("issue46.json"),
        )


def _run_actual_main(
    repo: Path,
    spec_path: Path,
    output_path: Path,
) -> int:
    return campaign_module.main(
        [
            "--spec",
            str(spec_path),
            "--cwd",
            str(repo),
            "--output-json",
            str(output_path),
        ]
    )


def _run_plan_main(repo: Path, spec_path: Path, *extra: str) -> int:
    return campaign_module.main(
        [
            "--spec",
            str(spec_path),
            "--cwd",
            str(repo),
            "--plan-only",
            *extra,
        ]
    )


def test_plan_only_binds_frozen_sources_shared_hooks_and_exact_matrix_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    expected_commit = _git(repo, "rev-parse", "HEAD")
    expected_spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    expected_hooks_sha = hashlib.sha256(
        (repo / HOOKS_CONFIG_SOURCE).read_bytes()
    ).hexdigest()
    hardware_calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: hardware_calls.append("run"),
    )

    assert _run_plan_main(repo, spec_path) == 0

    plan = json.loads(capsys.readouterr().out)
    assert hardware_calls == []
    assert plan["campaign_spec_sha256"] == expected_spec_sha
    assert plan["hardware_hooks_config_sha256"] == expected_hooks_sha
    assert plan["source_git_commit"] == expected_commit
    assert plan["context_matrix_tokens"] == [4_096, 32_768, 65_536, 131_072]
    assert plan["workload_subprocess_timeout_seconds"] == 43_200
    assert plan["qwen_control_timeout_seconds"] == 300
    assert plan["subprocess_termination_grace_seconds"] == 30
    assert plan["legacy_exclusive_lane_lock"] == "/tmp/mtplx-gpu-exclusive.lock"
    assert plan["legacy_exclusive_lane_wait_seconds"] == 0
    assert len(plan["schedule"]) == 16
    for command in (
        plan["artifact_verify_command"],
        plan["probe_command"],
        plan["quality_command"],
        plan["arm_command_template"],
    ):
        option_index = command.index("--hooks-config")
        assert command[option_index + 1] == HOOKS_CONFIG_SOURCE


def test_plan_only_rejects_untracked_command_source_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ignored_source = "benchmarks/ignored_probe.py"

    def use_ignored_probe(spec: dict[str, object]) -> None:
        spec["probe_command"] = [
            "uv",
            "run",
            "python",
            ignored_source,
            "--hooks-config",
            HOOKS_CONFIG_SOURCE,
            "--json",
        ]

    repo, spec_path = _create_campaign_repo(tmp_path, mutate_spec=use_ignored_probe)
    ignored_path = repo / ignored_source
    ignored_path.write_text("# ignored and untracked\n", encoding="utf-8")
    (repo / ".git" / "info" / "exclude").write_text(
        f"/{ignored_source}\n",
        encoding="utf-8",
    )
    hardware_calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: hardware_calls.append("run"),
    )

    with pytest.raises(BenchmarkGateError, match=r"command source.*tracked"):
        _run_plan_main(repo, spec_path)

    assert hardware_calls == []


def test_plan_only_rejects_split_hardware_hooks_identity_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate = "benchmarks/specs/alternate-hooks.json"

    def split_hooks(spec: dict[str, object]) -> None:
        quality = spec["quality_command"]
        assert isinstance(quality, list)
        quality[quality.index(HOOKS_CONFIG_SOURCE)] = alternate

    repo, spec_path = _create_campaign_repo(tmp_path, mutate_spec=split_hooks)
    alternate_path = repo / alternate
    alternate_path.write_text('{"frozen":"alternate"}\n', encoding="utf-8")
    _git(repo, "add", "--", alternate)
    _git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "track alternate hooks",
    )
    hardware_calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: hardware_calls.append("run"),
    )

    with pytest.raises(BenchmarkGateError, match="same hardware hooks config"):
        _run_plan_main(repo, spec_path)

    assert hardware_calls == []


def test_plan_only_rejects_dirty_frozen_spec_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["repetitions"] = 4
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    hardware_calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: hardware_calls.append("run"),
    )

    with pytest.raises(BenchmarkGateError, match=r"campaign spec.*HEAD"):
        _run_plan_main(repo, spec_path)

    assert hardware_calls == []


def test_plan_only_rejects_output_json_to_prevent_stale_acceptance_artifacts(
    tmp_path: Path,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "stale.json"
    output_path.write_text('{"status":"passed","stale":true}\n', encoding="utf-8")

    with pytest.raises(BenchmarkGateError, match="output-json.*plan-only"):
        _run_plan_main(repo, spec_path, "--output-json", str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8"))["stale"] is True


def test_actual_run_binds_clean_tracked_spec_and_command_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    expected_commit = _git(repo, "rev-parse", "HEAD")
    expected_spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    expected_hooks_sha = hashlib.sha256(
        (repo / HOOKS_CONFIG_SOURCE).read_bytes()
    ).hexdigest()
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_run_spec(spec: dict[str, object], *, cwd: Path) -> dict[str, object]:
        calls.append((cwd, spec))
        return {
            "schema": "mtplx-hy3-dynamic-memory-campaign-v1",
            "status": "passed",
        }

    monkeypatch.setattr(campaign_module, "run_spec", fake_run_spec)

    assert _run_actual_main(repo, spec_path, output_path) == 0

    stdout_result = json.loads(capsys.readouterr().out)
    file_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_result == file_result
    assert stdout_result["campaign_spec_sha256"] == expected_spec_sha
    assert stdout_result["hardware_hooks_config_sha256"] == expected_hooks_sha
    assert stdout_result["source_git_commit"] == expected_commit
    assert calls and calls[0][0] == repo.resolve()


def test_actual_run_rejects_an_ignored_untracked_hardware_hooks_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path, track_hooks_config=False)
    (repo / ".git" / "info" / "exclude").write_text(
        f"/{HOOKS_CONFIG_SOURCE}\n",
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    output_path = tmp_path / "campaign.json"
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"hardware hooks config.*tracked"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_rejects_commands_that_reference_different_hooks_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate = "benchmarks/specs/alternate-hooks.json"

    def split_hooks_configs(spec: dict[str, object]) -> None:
        probe = spec["probe_command"]
        assert isinstance(probe, list)
        probe[probe.index(HOOKS_CONFIG_SOURCE)] = alternate

    repo, spec_path = _create_campaign_repo(tmp_path, mutate_spec=split_hooks_configs)
    alternate_path = repo / alternate
    alternate_path.write_text('{"frozen":"alternate"}\n', encoding="utf-8")
    _git(repo, "add", "--", alternate)
    _git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "track alternate hooks config",
    )
    output_path = tmp_path / "campaign.json"
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"same hardware hooks config"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_rechecks_hardware_hooks_config_after_the_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    source_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        campaign_module,
        "_require_clean_source",
        lambda _repo: source_commit,
    )

    def mutate_hooks_config(
        _spec: dict[str, object],
        *,
        cwd: Path,
    ) -> dict[str, object]:
        (cwd / HOOKS_CONFIG_SOURCE).write_text(
            '{"frozen":"changed-during-run"}\n',
            encoding="utf-8",
        )
        return {"status": "passed"}

    monkeypatch.setattr(campaign_module, "run_spec", mutate_hooks_config)

    with pytest.raises(BenchmarkGateError, match=r"hardware hooks config.*HEAD"):
        _run_actual_main(repo, spec_path, output_path)

    assert not output_path.exists()


def test_actual_run_invalidates_stale_output_before_provenance_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path, track_spec=False)
    output_path = tmp_path / "campaign.json"
    output_path.write_text('{"status":"passed","stale":true}\n', encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"campaign spec.*tracked"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_writes_rejection_evidence_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    output_path.write_text('{"status":"passed","stale":true}\n', encoding="utf-8")
    rejected = {
        "schema": "mtplx-hy3-dynamic-memory-campaign-v1",
        "status": "rejected",
        "acceptance": {"passed": False, "rejection_reasons": ["gate failed"]},
    }
    monkeypatch.setattr(campaign_module, "run_spec", lambda *_args, **_kwargs: rejected)

    assert _run_actual_main(repo, spec_path, output_path) == 2

    stdout_result = json.loads(capsys.readouterr().out)
    file_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_result == file_result
    assert file_result["status"] == "rejected"
    assert file_result["acceptance"]["passed"] is False
    assert "stale" not in file_result


def test_actual_run_rejects_an_untracked_campaign_spec_before_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path, track_spec=False)
    output_path = tmp_path / "campaign.json"
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"campaign spec.*tracked"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_rejects_a_dirty_tracked_command_source_before_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    (repo / PROBE_SOURCE).write_text("# dirty command source\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"clean source worktree|dirty"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_rejects_an_ignored_untracked_command_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ignored_source = "benchmarks/ignored_probe.py"

    def use_ignored_probe(spec: dict[str, object]) -> None:
        spec["probe_command"] = ["uv", "run", "python", ignored_source]

    repo, spec_path = _create_campaign_repo(tmp_path, mutate_spec=use_ignored_probe)
    ignored_path = repo / ignored_source
    ignored_path.write_text("# ignored and untracked\n", encoding="utf-8")
    (repo / ".git" / "info" / "exclude").write_text(
        f"/{ignored_source}\n",
        encoding="utf-8",
    )
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    output_path = tmp_path / "campaign.json"
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(
        BenchmarkGateError, match=r"command source.*tracked|tracked.*source"
    ):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


@pytest.mark.parametrize(
    "command_field",
    ("probe", "arm", "quality", "qwen"),
)
def test_actual_run_rejects_arbitrary_non_source_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_field: str,
) -> None:
    def insert_arbitrary_command(spec: dict[str, object]) -> None:
        arbitrary = ["/bin/echo", "not-a-campaign-source"]
        if command_field == "probe":
            spec["probe_command"] = arbitrary
        elif command_field == "arm":
            spec["arm_command_template"] = arbitrary
        elif command_field == "quality":
            spec["quality_command"] = arbitrary
        else:
            qwen = spec["qwen"]
            assert isinstance(qwen, dict)
            qwen["capture_command"] = arbitrary

    repo, spec_path = _create_campaign_repo(
        tmp_path,
        mutate_spec=insert_arbitrary_command,
    )
    output_path = tmp_path / "campaign.json"
    calls: list[object] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(
        BenchmarkGateError,
        match=r"command source|tracked.*source|Python script",
    ):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
    assert not output_path.exists()


def test_actual_run_rechecks_clean_provenance_after_the_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"

    def dirty_during_workload(
        _spec: dict[str, object],
        *,
        cwd: Path,
    ) -> dict[str, object]:
        (cwd / ARM_SOURCE).write_text("# changed during campaign\n", encoding="utf-8")
        return {
            "schema": "mtplx-hy3-dynamic-memory-campaign-v1",
            "status": "passed",
        }

    monkeypatch.setattr(campaign_module, "run_spec", dirty_during_workload)

    with pytest.raises(BenchmarkGateError, match=r"clean source worktree|dirty"):
        _run_actual_main(repo, spec_path, output_path)

    assert not output_path.exists()


def test_actual_run_executes_the_same_spec_bytes_it_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    frozen_bytes = spec_path.read_bytes()
    original_load = campaign_module._load_spec
    executed_seeds: list[int] = []

    def transient_replacement(path: Path) -> dict[str, object]:
        replacement = _default_spec()
        replacement["bootstrap_seed"] = 999
        path.write_text(json.dumps(replacement), encoding="utf-8")
        try:
            return original_load(path)
        finally:
            path.write_bytes(frozen_bytes)

    def fake_run_spec(spec: dict[str, object], *, cwd: Path) -> dict[str, object]:
        del cwd
        executed_seeds.append(int(spec["bootstrap_seed"]))
        return {"status": "passed"}

    monkeypatch.setattr(campaign_module, "_load_spec", transient_replacement)
    monkeypatch.setattr(campaign_module, "run_spec", fake_run_spec)

    assert _run_actual_main(repo, spec_path, output_path) == 0

    result = json.loads(capsys.readouterr().out)
    assert executed_seeds == [46]
    assert result["campaign_spec_sha256"] == hashlib.sha256(frozen_bytes).hexdigest()


def test_actual_run_rejects_inline_code_with_a_tracked_script_decoy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def insert_inline_probe(spec: dict[str, object]) -> None:
        spec["probe_command"] = [
            "python",
            "-c",
            "print('fabricated campaign evidence')",
            PROBE_SOURCE,
        ]

    repo, spec_path = _create_campaign_repo(
        tmp_path,
        mutate_spec=insert_inline_probe,
    )
    output_path = tmp_path / "campaign.json"
    calls: list[str] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"Python script|command source"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []


def test_actual_run_rejects_uv_executable_with_a_tracked_script_decoy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def insert_decoy_probe(spec: dict[str, object]) -> None:
        spec["probe_command"] = [
            "uv",
            "run",
            "/bin/echo",
            "python",
            PROBE_SOURCE,
        ]

    repo, spec_path = _create_campaign_repo(tmp_path, mutate_spec=insert_decoy_probe)
    output_path = tmp_path / "campaign.json"
    calls: list[str] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=r"Python script|command source"):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []


@pytest.mark.parametrize(
    ("index_flag", "expected_marker"),
    (("--assume-unchanged", "assume-unchanged"), ("--skip-worktree", "skip-worktree")),
)
def test_actual_run_rejects_hidden_index_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
    expected_marker: str,
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    _git(repo, "update-index", index_flag, PROBE_SOURCE)
    (repo / PROBE_SOURCE).write_text("# hidden source replacement\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    output_path = tmp_path / "campaign.json"
    calls: list[str] = []
    monkeypatch.setattr(
        campaign_module,
        "run_spec",
        lambda *_args, **_kwargs: calls.append("run") or {"status": "passed"},
    )

    with pytest.raises(BenchmarkGateError, match=expected_marker):
        _run_actual_main(repo, spec_path, output_path)

    assert calls == []
