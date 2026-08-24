from __future__ import annotations

from pathlib import Path
import tomllib


DFLASH_MLX_PIN = (
    "dflash-mlx @ "
    "git+https://github.com/davidtai/dflash-mlx.git@"
    "dc442a608bf062e118901ee4e43a436ef083a11a"
)


def test_dflash2_dependency_is_immutably_pinned() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["optional-dependencies"]["competitors"] == [
        DFLASH_MLX_PIN
    ]
