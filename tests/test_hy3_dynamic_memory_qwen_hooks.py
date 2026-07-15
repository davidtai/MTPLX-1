from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mtplx.benchmarks.hy3_dynamic_memory_qwen as qwen_module
import scripts.run_issue30_starvation_attribution as issue30_module
from scripts.run_issue30_starvation_attribution import QwenState


_OWNER_ENV = {
    "MTPLX_ISSUE46_CAMPAIGN_OWNER_TOKEN": "a" * 64,
    "MTPLX_ISSUE46_CAMPAIGN_PID": str(os.getpid()),
}


def test_tracked_script_can_import_issue30_guard_from_benchmarks_directory() -> None:
    script = (
        Path(__file__).resolve().parent.parent / "benchmarks" / "hy3_qwen_isolation.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Exact-state Qwen isolation" in completed.stdout


def test_capture_emits_exact_loaded_and_models_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = QwenState(
        loaded=True,
        models=("mtplx-qwen36-27b-optimized-speed",),
    )
    monkeypatch.setattr(qwen_module, "_capture_qwen", lambda **_kwargs: captured)

    assert qwen_module.main(["capture"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "loaded": True,
        "models": ["mtplx-qwen36-27b-optimized-speed"],
    }


def test_unload_consumes_captured_state_and_proves_qwen_stopped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = QwenState(
        loaded=True,
        models=("mtplx-qwen36-27b-optimized-speed",),
    )
    calls: list[object] = []
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(state._asdict())))
    monkeypatch.setattr(
        qwen_module,
        "_stop_qwen",
        lambda observed, **_kwargs: calls.append(("stop", observed)),
    )
    monkeypatch.setattr(
        qwen_module,
        "_capture_qwen",
        lambda **_kwargs: QwenState(loaded=False, models=()),
    )

    assert qwen_module.main(["unload"]) == 0

    assert calls == [("stop", state)]
    assert json.loads(capsys.readouterr().out) == {
        "loaded": False,
        "models": [],
        "unloaded": True,
    }


def test_restore_and_verify_require_exact_captured_model_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = QwenState(
        loaded=True,
        models=("mtplx-qwen36-27b-optimized-speed",),
    )
    calls: list[object] = []
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(state._asdict())))
    monkeypatch.setattr(
        qwen_module,
        "_restore_qwen",
        lambda observed, **_kwargs: calls.append(("restore", observed)),
    )
    monkeypatch.setattr(qwen_module, "_capture_qwen", lambda **_kwargs: state)

    assert qwen_module.main(["restore"]) == 0
    assert calls == [("restore", state)]
    assert json.loads(capsys.readouterr().out) == {
        "loaded": True,
        "models": ["mtplx-qwen36-27b-optimized-speed"],
        "restored": True,
    }

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(state._asdict())))
    monkeypatch.setattr(
        qwen_module,
        "_capture_qwen",
        lambda **_kwargs: QwenState(loaded=False, models=()),
    )
    assert qwen_module.main(["verify"]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "differs from the captured state" in streams.err


def test_lane_acquire_keeps_issue30_diagnostics_off_json_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}

    def acquire(*, poll_seconds: float, exclude_pids: tuple[int, ...]) -> None:
        assert poll_seconds == 2.0
        assert exclude_pids == (1234,)
        print("waiting for existing benchmark")
        lane.mkdir()

    monkeypatch.setattr(qwen_module, "EXCLUSIVE_LANE", lane)
    monkeypatch.setattr(qwen_module, "_acquire_lane", acquire)
    monkeypatch.setattr(qwen_module, "_campaign_owner_identity", lambda: owner)

    assert qwen_module.main(["acquire"]) == 0

    streams = capsys.readouterr()
    assert json.loads(streams.out) == {
        "acquired": True,
        "lane": str(lane),
        "owner": owner,
    }
    assert "waiting for existing benchmark" in streams.err
    assert qwen_module.main(["release"]) == 0


def test_lane_release_requires_the_acquiring_campaign_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    monkeypatch.setattr(qwen_module, "EXCLUSIVE_LANE", lane)
    monkeypatch.setattr(
        qwen_module,
        "_acquire_lane",
        lambda **_kwargs: lane.mkdir(),
    )
    monkeypatch.setattr(qwen_module, "_campaign_owner_identity", lambda: owner)

    assert qwen_module.main(["acquire"]) == 0
    capsys.readouterr()
    assert json.loads((lane / "issue46-owner.json").read_text()) == owner

    monkeypatch.setattr(
        qwen_module,
        "_campaign_owner_identity",
        lambda: {**owner, "campaign_pid": 9999},
    )
    assert qwen_module.main(["release"]) == 2
    assert lane.is_dir()
    assert "owner" in capsys.readouterr().err

    monkeypatch.setattr(qwen_module, "_campaign_owner_identity", lambda: owner)
    assert qwen_module.main(["release"]) == 0
    assert not lane.exists()


def test_lane_acquire_identifies_owner_before_claiming_lane(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    acquire_calls: list[float] = []
    monkeypatch.setattr(
        qwen_module,
        "_campaign_owner_identity",
        lambda: (_ for _ in ()).throw(qwen_module.QwenIsolationError("no parent")),
    )
    monkeypatch.setattr(
        qwen_module,
        "_acquire_lane",
        lambda *, poll_seconds, **_kwargs: acquire_calls.append(poll_seconds),
    )

    assert qwen_module.main(["acquire"]) == 2

    assert acquire_calls == []
    assert "no parent" in capsys.readouterr().err


def test_lane_acquire_removes_partial_owner_file_after_durable_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lane = tmp_path / "exclusive-lane"
    owner = {"uid": 501, "campaign_pid": 1234, "owner_token_sha256": "c" * 64}
    monkeypatch.setattr(qwen_module, "EXCLUSIVE_LANE", lane)
    monkeypatch.setattr(
        qwen_module,
        "_acquire_lane",
        lambda **_kwargs: lane.mkdir(),
    )
    monkeypatch.setattr(qwen_module, "_campaign_owner_identity", lambda: owner)
    monkeypatch.setattr(
        qwen_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    assert qwen_module.main(["acquire"]) == 2

    assert not lane.exists()
    assert "fsync failed" in capsys.readouterr().err


def test_issue30_guard_excludes_the_stable_campaign_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(issue30_module.os, "getpid", lambda: 10)
    monkeypatch.setattr(issue30_module.os, "getppid", lambda: 11)
    monkeypatch.setattr(
        issue30_module,
        "_run_text",
        lambda *_args, **_kwargs: "10\n11\n41\n42",
    )

    assert issue30_module._matching_processes(exclude_pids=(41,)) == (42,)


def test_uv_hook_processes_share_one_explicit_campaign_owner(
    tmp_path: Path,
) -> None:
    lane = tmp_path / "exclusive-lane"
    code = """
import sys
from pathlib import Path
import mtplx.benchmarks.hy3_dynamic_memory_qwen as qwen
import scripts.run_issue30_starvation_attribution as issue30
lane = Path(sys.argv[2])
qwen.EXCLUSIVE_LANE = lane
issue30.EXCLUSIVE_LANE = lane
raise SystemExit(qwen.main([sys.argv[1]]))
"""
    env = {**os.environ, **_OWNER_ENV}

    acquired = subprocess.run(
        ["uv", "run", "python", "-c", code, "acquire", str(lane)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert acquired.returncode == 0, acquired.stderr
    released = subprocess.run(
        ["uv", "run", "python", "-c", code, "release", str(lane)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert released.returncode == 0, released.stderr
    assert json.loads(acquired.stdout)["owner"] == json.loads(released.stdout)["owner"]
    assert not lane.exists()


def test_issue30_exclusive_guard_recognizes_issue46_campaign_processes() -> None:
    assert (
        "benchmark_hy3_dynamic_memory.py" in issue30_module.BENCHMARK_PROCESS_PATTERNS
    )
    assert (
        "observe_hy3_dynamic_memory_arm.py" in issue30_module.BENCHMARK_PROCESS_PATTERNS
    )
    assert "probe_hy3_direct_cache.py" in issue30_module.BENCHMARK_PROCESS_PATTERNS


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_qwen_timeouts_reject_nonfinite_values(
    invalid: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert qwen_module.main(["capture", f"--api-timeout-seconds={invalid}"]) == 2

    streams = capsys.readouterr()
    assert streams.out == ""
    assert "must be positive" in streams.err
