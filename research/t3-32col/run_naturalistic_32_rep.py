#!/usr/bin/env python3
"""Naturalistic short-prompt arm driver for the T3 **32-envelope column**
(2026-07-22), one rep per invocation. Sibling of
research/t3-80col/run_naturalistic_80_rep.py, retargeted at the
`hy3-oq2e-rq4-32-cachelru` preset (ZERO islands, cache-heavy LRU -- David's
<=64 GiB no-islands directive) instead of the islands-69 `-80` preset --
otherwise identical mechanics.

`hy3-oq2e` has no `--prompt-tail` CLI flag (MODEL_SPECS["hy3-oq2e"]["prompt_tail"]
is hardcoded None), so this calls the library entry point directly
(`benchmark_q2_mtp_depth_matrix.main`, which internally calls
`run_depth_matrix()`) and monkeypatches `_requests_from_args` -- the same
helper `main()` itself calls -- to inject `prompt_tail_text` into the
hy3-oq2e request after it's built.

CONTEXT_TOKENS=265 (264 chat-templated tail tokens + 1 structurally-required
filler token) is the SAME derivation as every other T3 sibling -- direct-
probed once, reused as-is here since the tail content and chat-template
plumbing are identical; only the preset differs.

Usage (never invoked directly except by run_32col_inner.sh, which already
holds the flock + has qwen stopped):
    PYTHONPATH=<worktree> python3 \
        research/t3-32col/run_naturalistic_32_rep.py \
        --output-json evals/tier2/t3_32col_naturalistic_repN.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_q2_mtp_depth_matrix as bench  # noqa: E402

TAIL_PATH = REPO_ROOT / "research" / "naturalistic_prompt_v1.txt"
TARGET_MODEL = "hy3-oq2e"
CONTEXT_TOKENS = 265  # tail (264 chat-templated tokens) + 1 structural filler token
PRESET = "hy3-oq2e-rq4-32-cachelru"
OUTPUT_TOKENS = 256
HY3_DEPTHS = "1"

_orig_requests_from_args = bench._requests_from_args


def _patched_requests_from_args(args):
    requests = _orig_requests_from_args(args)
    tail_text = TAIL_PATH.read_text(encoding="utf-8")
    patched = False
    for request in requests:
        if request.get("model") == TARGET_MODEL:
            request["prompt_tail_text"] = tail_text
            request["prompt_tail"] = TAIL_PATH.resolve()
            patched = True
    if not patched:
        raise RuntimeError(
            f"naturalistic driver expected model {TARGET_MODEL!r} in the "
            f"preset's request list, found: {[r.get('model') for r in requests]}"
        )
    return requests


bench._requests_from_args = _patched_requests_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    ns = parser.parse_args()

    argv = [
        "--preset", PRESET,
        "--contexts", str(CONTEXT_TOKENS),
        "--output-tokens", str(OUTPUT_TOKENS),
        "--hy3-depths", HY3_DEPTHS,
        "--output-json", str(ns.output_json),
    ]
    print(f"[naturalistic-32] invoking bench.main({argv!r})", file=sys.stderr)
    return bench.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
