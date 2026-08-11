from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_venv_smoke_installs_numpy_before_running_the_cli() -> None:
    script = (ROOT / "scripts" / "fresh_venv_smoke.sh").read_text(encoding="utf-8")

    numpy_install = '"$VENV/bin/python" -m pip install "numpy>=2"'
    cli_smoke = '"$VENV/bin/mtplx" --help'

    assert numpy_install in script
    assert script.index(numpy_install) < script.index(cli_smoke)
