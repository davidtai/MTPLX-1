#!/usr/bin/env python3
"""JSON-only Qwen isolation command for the issue #46 hardware campaign."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.hy3_dynamic_memory_qwen import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
