#!/usr/bin/env bash
# OUTER launcher for CONTINUATION window 3: the K3-exact patch lane, 3 reps.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_window3] CPU-only admission preflight re-check (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-64-ab/preflight.py \
  > "$WT/evals/tier2/t3_64_ab_admission_preflight_window3.json" \
  2> "$WT/evals/tier2/t3_64_ab_admission_preflight_window3.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_window3] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_64_ab_admission_preflight_window3.log" >&2
  exit 2
fi
echo "[run_window3] preflight OK" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 3600 \
  -- \
  bash "$WT/research/t3-64-ab/run_window3_patch_inner.sh"
