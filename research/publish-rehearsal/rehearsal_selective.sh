#!/usr/bin/env bash
# OUTER launcher: HF-publish clean-room rehearsal (selective clone, 64 env).
# Launch via Bash run_in_background:true, STANDALONE -- no chaining, no
# shell '&', no Bash timeout parameter. Short window (~10-20 min).
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[publish-rehearsal] wired-knob (recorded): $(sysctl -n iogpu.wired_limit_mb 2>/dev/null)" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/publish-rehearsal/rehearsal_selective_inner.sh"
