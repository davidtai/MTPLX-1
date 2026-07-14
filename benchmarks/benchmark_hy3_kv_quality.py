#!/usr/bin/env python3
"""Run the standalone Hy3 BF16-control versus Q4-KV quality gate."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.hy3_kv_quality import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
