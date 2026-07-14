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
QWEN_SOURCE = "benchmarks/hy3_qwen_isolation.py"
SPEC_SOURCE = "benchmarks/specs/issue46.json"
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
        "qwen": qwen,
        "probe_command": ("probe",),
        "arm_command_template": ("arm",),
        "repetitions": 2,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 46,
    }
    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_json_subprocess(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> dict[str, object]:
        action = command[0]
        env = kwargs.get("env")
        assert env is None or isinstance(env, dict)
        calls.append((action, env))
        if action == "capture":
            return {"loaded": False, "models": []}
        if action == "verify":
            return {"restored": True}
        return {"ok": True}

    monkeypatch.setattr(campaign_module.secrets, "token_hex", lambda _size: "b" * 64)
    monkeypatch.setattr(campaign_module, "run_json_subprocess", fake_json_subprocess)
    monkeypatch.setattr(
        campaign_module,
        "run_subprocess_campaign",
        lambda **_kwargs: {"status": "passed"},
    )

    assert campaign_module.run_spec(spec, cwd=tmp_path) == {"status": "passed"}

    qwen_calls = calls
    assert [action for action, _env in qwen_calls] == [
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
    assert all(env == expected_env for _action, env in qwen_calls)


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
        "probe_command": ["uv", "run", "python", PROBE_SOURCE, "--json"],
        "arm_command_template": [
            "uv",
            "run",
            "python",
            ARM_SOURCE,
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
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in (PROBE_SOURCE, ARM_SOURCE, QWEN_SOURCE):
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

    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Issue 46 Test")
    _git(repo, "config", "user.email", "issue46@example.invalid")
    tracked = [PROBE_SOURCE, ARM_SOURCE, QWEN_SOURCE]
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


def test_actual_run_binds_clean_tracked_spec_and_command_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, spec_path = _create_campaign_repo(tmp_path)
    output_path = tmp_path / "campaign.json"
    expected_commit = _git(repo, "rev-parse", "HEAD")
    expected_spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
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
    assert stdout_result["source_git_commit"] == expected_commit
    assert calls and calls[0][0] == repo.resolve()


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
    ("probe", "arm", "qwen"),
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
