#!/usr/bin/env bash
# OUTER launcher: single-rep recovery for 64x64k rep3 (2026-07-22).
# The first attempt died with the 64col wrapper's EPERM teardown at rep3
# start (receipt froze partial, AR row only). Precedent: single-rep
# recovery windows (kselect bf16 reps 2-3). Queues on the flock behind the
# smallcode-redo window. Launch via Bash run_in_background:true,
# STANDALONE -- no chaining, no shell '&', no Bash timeout parameter.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[64col-rep3-recovery] wired-knob (recorded): $(sysctl -n iogpu.wired_limit_mb 2>/dev/null)" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-64col/run_64col_rep3_recovery_inner.sh"
