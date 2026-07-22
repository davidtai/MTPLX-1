#!/usr/bin/env bash
# OUTER launcher: T3 80-column RESUME guarded window (2026-07-22).
# Identical protocol to run_80col.sh (see its header for arms, overrides,
# sign-offs) but exec's run_80col_resume_inner.sh, which runs ONLY the reps
# the killed first window never finished: 16k reps 2-3, 64k reps 1-3,
# naturalistic reps 1-3. Launched from the MAIN session so a driver stop
# cannot take the window down (2026-07-22 lesson, x3).
# Launch via Bash run_in_background:true, STANDALONE -- no chaining, no
# shell '&', no Bash timeout parameter.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_80col_resume] wired-knob (recorded): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_80col_resume] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_80col_wired_knob.txt" >&2

echo "[run_80col_resume] CPU-only admission preflight (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-80col/preflight.py \
  --output "$WT/evals/tier2/t3_80col_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_80col_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_80col_resume] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_80col_admission_preflight.log" >&2
  exit 2
fi
echo "[run_80col_resume] preflight OK" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-80col/run_80col_resume_inner.sh"
