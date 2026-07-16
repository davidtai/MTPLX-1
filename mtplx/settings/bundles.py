from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


_SCALAR_TYPES = (bool, int, float, str)


def load_settings_bundle(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    with resolved.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"settings"} or not isinstance(payload["settings"], dict):
        raise ValueError(f"{resolved}: settings bundles may contain only [settings]")
    settings: dict[str, Any] = {}
    for name, value in payload["settings"].items():
        if not isinstance(name, str) or not isinstance(value, _SCALAR_TYPES):
            raise ValueError(
                f"{resolved}: settings must use string keys and scalar values"
            )
        settings[name] = value
    return settings
