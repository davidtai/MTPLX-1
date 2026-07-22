#!/usr/bin/env bash
# OUTER launcher: T3 real-prefill PILOT guarded window (2026-07-22).
#
# Mission (HANDOFF-hy3-matrix-restart.md work-queue item 2): validate the
# harness's --contexts real-prefill mechanics end to end with ONE 1-rep,
# cheap-preset window BEFORE the expensive 3-rep K-selection window at
# efficient-88x16k. This is mechanics validation, NOT a cell of record --
# do not cite its numbers as a performance result.
#
# Preset `hy3-oq2e-rq4-32-cachelru` used AS-IS (zero islands, cache-heavy
# LRU, David's <=64 GiB directive). CPU-admission-verified: ADMIT margin
# +0.2147 GiB at kv=16640 (16384 context + 256 output)/kv_quant=q4, preset's
# own 39 GiB memory-limit, NO --memory-limit override needed (see research/
# t3-pilot-realprefill/preflight.py).
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all is exported inside
# run_pilot_inner.sh (in NO preset, must be set manually, read once per
# process -- see run_pilot_inner.sh header).
#
# CPU-only preflight MUST pass before the guarded window opens -- if it
# REJECTs, this script exits 2 and NO GPU work happens, no qwen stop, no
# flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

PRESET="hy3-oq2e-rq4-32-cachelru"
KV=16640

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_pilot] CPU-only admission preflight for $PRESET @ kv=$KV kv_quant=q4 (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-pilot-realprefill/preflight.py --kv "$KV" --kv-quant q4 \
  > "$WT/evals/tier2/pilot_realprefill_admission_preflight.json" \
  2> "$WT/evals/tier2/pilot_realprefill_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_pilot] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/pilot_realprefill_admission_preflight.log" >&2
  exit 2
fi
echo "[run_pilot] preflight OK; $PRESET ADMITs at kv=$KV kv_quant=q4" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 18000 \
  -- \
  bash "$WT/research/t3-pilot-realprefill/run_pilot_inner.sh"
