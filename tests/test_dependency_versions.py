from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_mlx_032_is_required_across_runtime_build_and_lock() -> None:
    project = _read_toml(ROOT / "pyproject.toml")
    runtime_mlx = [
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("mlx>")
    ]

    native_project = _read_toml(
        ROOT / "native_extensions" / "expert_io" / "pyproject.toml"
    )
    native_mlx = [
        requirement
        for requirement in native_project["build-system"]["requires"]
        if requirement.startswith("mlx==")
    ]

    lock = _read_toml(ROOT / "uv.lock")
    locked_mlx = [
        package["version"] for package in lock["package"] if package["name"] == "mlx"
    ]
    locked_mlx_metal = [
        package["version"]
        for package in lock["package"]
        if package["name"] == "mlx-metal"
    ]

    assert runtime_mlx == [
        "mlx>=0.32,<0.33; sys_platform == 'darwin' and platform_machine == 'arm64'"
    ]
    assert native_mlx == ["mlx==0.32.0"]
    assert locked_mlx == ["0.32.0"]
    assert locked_mlx_metal == ["0.32.0"]
