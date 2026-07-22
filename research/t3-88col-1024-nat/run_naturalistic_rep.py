#!/usr/bin/env python3
"""Naturalistic short-prompt arm driver (2026-07-22), one rep per invocation.

`hy3-oq2e` has no `--prompt-tail` CLI flag (MODEL_SPECS["hy3-oq2e"]["prompt_tail"]
is hardcoded None, and none of the three per-model `--*-prompt-tail` flags
name oq2e). This script calls the library entry point directly
(`benchmark_q2_mtp_depth_matrix.main`, which internally calls
`run_depth_matrix()`) and monkeypatches `_requests_from_args` -- the same
helper `main()` itself calls -- to inject `prompt_tail_text` into the
hy3-oq2e request after it's built, changing NOTHING else about the CLI's own
argument parsing / preset application / advisory / checkpoint / exit-code
plumbing. `_normalized_request` (scripts/benchmark_q2_mtp_depth_matrix.py
:2574-2592) already threads `prompt_tail_text` for any model; this was
confirmed by direct probe (see NOTES below), not assumed.

CONTEXT-TOKEN MECHANICS (probed CPU-only before this script existed, zero
GPU touch, zero model load -- only the small tokenizer.json + chat template
were read from the oq2e model_root):

  research/naturalistic_prompt_v1.txt chat-templates (prompt_style=
  "coding-agent" -> chat format, enable_thinking=False, same options
  _run_depth_matrix_impl always uses) to exactly 264 tokens.

  The plan text handed down for this window said "set contexts=N where N
  is the tail's exact token count, so it truncates cleanly with NO filler
  (the `len(tail_ids) >= context_tokens` branch, prefill_bench.py:196-212)".
  That branch was direct-probed at contexts=264 (== tail length) and DOES
  produce a clean truncation with zero filler tokens -- but it ALSO stamps
  prompt_release_valid=False and prompt_tail_preserved=False (prefill_bench
  .py:208,211), and `_run_depth_matrix_impl` hard-gates on BOTH of those
  being true (scripts/benchmark_q2_mtp_depth_matrix.py:2992-3004) -- it
  raises BenchmarkGateError inside run_depth_matrix() for the exact-N case,
  which would have failed mid-window after a real model load. Probed at
  contexts=265 (== tail length + 1) instead: this takes the OTHER branch
  (prefill_bench.py:219-289, filler injection) with filler_target=1 -- one
  stray token before the tail is structurally unavoidable through this
  harness entry point, and the metadata reads prompt_policy=
  "realistic_programming_v1", prompt_format="chat", prompt_release_valid=
  True, prompt_tail_preserved=True, prompt_filler_tokens=1 -- a normal,
  fully release-valid row, distinguished from the long-code matrix cells
  only by the ~99.6%-non-code tail content and much shorter length. This
  script therefore uses CONTEXT_TOKENS=265, not 264. (Both figures are
  recorded in the receipt; the validator checks token_count==265 and
  reports, rather than assumes, prompt_release_valid -- per instruction,
  since this cell's true shape differs slightly from the plan text.)

Usage (never invoked directly except by run_88col_1024_nat_inner.sh, which
already holds the flock + has qwen stopped):
    PYTHONPATH=<worktree> python3 \
        research/t3-88col-1024-nat/run_naturalistic_rep.py \
        --output-json evals/tier2/t3_88col_naturalistic_repN.json
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
PRESET = "hy3-oq2e-rq4-88e"
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
    print(f"[naturalistic] invoking bench.main({argv!r})", file=sys.stderr)
    return bench.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
