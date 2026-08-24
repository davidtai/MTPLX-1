"""MTPLX: native Qwen3.6 MTP experiments on MLX."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from .version import DISPLAY_VERSION, __version__

__all__ = ["MTPLXRuntime", "load", "__version__", "DISPLAY_VERSION"]


def _bootstrap_dflash2_command_buffer_environment(argv: list[str] | None = None) -> bool:
    """Latch the measured Turbo/DFlash environment before MLX imports."""

    arguments = list(sys.argv if argv is None else argv)
    model_value: str | None = None
    for index, argument in enumerate(arguments):
        if argument == "--model" and index + 1 < len(arguments):
            model_value = arguments[index + 1]
            break
        if argument.startswith("--model="):
            model_value = argument.split("=", 1)[1]
            break
    if not model_value:
        return False
    model_path = Path(model_value).expanduser()
    if not (model_path / "mtplx_dflash2.json").is_file():
        return False
    from .profiles import apply_profile_env

    apply_profile_env("turbo")
    # The external DFlash draft replaces every native/source MTP proposal path.
    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    # The final 16K gate retired row 53. MLX reads these once, so the measured
    # stock policy must be selected before any MLX import.
    os.environ.pop("MLX_MAX_MB_PER_BUFFER", None)
    os.environ.pop("MLX_MAX_OPS_PER_BUFFER", None)
    return True


_bootstrap_dflash2_command_buffer_environment()


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runtime import MTPLXRuntime, load

        exports = {"MTPLXRuntime": MTPLXRuntime, "load": load}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
