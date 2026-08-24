from __future__ import annotations

from pathlib import Path
import tomllib


DFLASH_MLX_PIN = (
    "dflash-mlx @ "
    "git+https://github.com/davidtai/dflash-mlx.git@"
    "dc49f5ac9a734ad1b69b7100043d2cbbf945b045"
)


def test_dflash2_dependency_is_immutably_pinned() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["optional-dependencies"]["competitors"] == [
        DFLASH_MLX_PIN
    ]
