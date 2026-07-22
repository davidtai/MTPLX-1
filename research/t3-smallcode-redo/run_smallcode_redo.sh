#!/usr/bin/env bash
# OUTER launcher: T3 small-code arm-of-record guarded window (2026-07-22).
# David's correction (live): the per-envelope "small naturalistic prompt"
# arm that already ran at 88/80/64 (t3_88col_naturalistic_*,
# t3_80col_naturalistic_*, t3_64col_naturalistic_*) was ALWAYS meant to be a
# SMALL CODE prompt, not prose -- this is a coding LLM, prose is out of
# distribution. Those prose arms are downgraded to "prose reference --
# incidental, not of record". This window runs the arm of record instead:
# contexts=320, output-tokens=256, K1 (--hy3-depths 1; AR row auto-runs
# alongside for free), bf16, greedy, the SAME release-valid
# realistic_programming_v1 builder every context cell already uses (plain
# `--contexts 320`, NO custom prompt_tail, NO library-call mechanics), 3
# reps each, against the THREE already-done envelopes, SERIALLY, in ONE
# guarded window (one qwen-stop/flock hold covering all nine process
# invocations):
#
# contexts=320: minimum viable N that preserves the standard coding tail
# (256 failed the preserve gate 2026-07-22 -- see research/t3-smallcode-redo/
# preflight.py and run_smallcode_redo_inner.sh headers for the full probe);
# David's "256 is fine" honored as close as the gate allows.
#
#   ARM 88e: hy3-oq2e-rq4-88e (islands 79, full residency).
#   ARM 80:  hy3-oq2e-rq4-80  (islands 69, 10 streamed).
#   ARM 64:  hy3-oq2e-rq4-64-cachelru (ZERO islands, cache-heavy LRU --
#            David's <=64 GiB no-islands directive).
#
# Memory-limit overrides (harness-derived by research/t3-smallcode-redo/
# preflight.py, never hand-derived): NONE needed for any of the three arms
# -- 320+256=576 tokens sits well under the shared 4096-token control point
# every preset already declares as its own default; all three ADMIT
# directly at their own declared limit (88e 96.0000 GiB margin +0.3706 GiB,
# 80 87.0000 GiB margin +0.0223 GiB, 64-cachelru 71.0000 GiB margin
# +0.3667 GiB). This admission math does not depend on contexts beyond the
# kv-sum bound check -- raising contexts 256->320 does not change these
# figures.
#
# Sign-off (David's rule 7): every limit here (96 / 87 / 71 GiB) sits well
# under the sub-85/90 GiB sign-off band (88e is the one exception already
# accepted at every sibling 88e window; its 96 GiB declared limit is
# unchanged here) and the 112 GiB wired knob -- no fresh ask needed.
#
# CPU-only preflight (research/t3-smallcode-redo/preflight.py) MUST pass
# (all three arms ADMIT) before this guarded window opens -- if it fails,
# this script exits 2 and NO GPU work happens, no qwen stop, no flock
# acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_smallcode_redo] wired-knob (recorded, not gated at this margin): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_smallcode_redo] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_smallcode_redo_wired_knob.txt" >&2

echo "[run_smallcode_redo] CPU-only admission preflight (zero GPU touch) for all three arms..." >&2
PYTHONPATH="$WT" "$PY" research/t3-smallcode-redo/preflight.py \
  --output "$WT/evals/tier2/t3_smallcode_redo_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_smallcode_redo_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_smallcode_redo] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_smallcode_redo_admission_preflight.log" >&2
  exit 2
fi
echo "[run_smallcode_redo] preflight OK; all three arms ADMIT" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-smallcode-redo/run_smallcode_redo_inner.sh"
