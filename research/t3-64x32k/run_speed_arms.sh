#!/usr/bin/env bash
# OUTER launcher: T3-pre 64x32k paired kv-mode arms (bf16 / kv8 / kv4),
# preset hy3-oq2e-rq4-64, --max-live-kv-tokens 32768, shared memory-limit
# override 79.75 GiB (85630910464 bytes).
#
# CPU-only preflight (research/t3-64x32k/preflight_kv_arms.py, the exact
# admission gate replica) MUST pass before the guarded window opens -- if it
# rejects, this script exits 2 and NO GPU work happens, no qwen stop, no
# flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a STANDALONE
# command (never chained, never shell `&`) -- box law: a chained window died
# to a hook timeout on 2026-07-21.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_speed_arms] CPU-only admission preflight (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-64x32k/preflight_kv_arms.py \
  > "$WT/evals/tier2/t3_64x32k_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_64x32k_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_speed_arms] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_64x32k_admission_preflight.log" >&2
  exit 2
fi
echo "[run_speed_arms] preflight OK; all three kv-mode arms ADMIT under the 79.75 GiB override" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-64x32k/run_speed_arms_inner.sh"
