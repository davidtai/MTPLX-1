#!/usr/bin/env bash
# OUTER launcher: T3 88-column completion guarded window (2026-07-22).
# Finishes the 88 envelope with its two remaining arms in ONE guarded window
# (one qwen-stop/flock hold covering six sequential model loads: 3 reps
# fresh-1024 + 3 reps naturalistic-~265):
#
#   ARM 1: hy3-oq2e-rq4-88e x contexts=1024 x output-tokens=256 x
#          hy3-depths=1 (+free AR), bf16 KV.
#   ARM 2: same preset/shape, contexts=265 (naturalistic non-code tail via
#          research/t3-88col-1024-nat/run_naturalistic_rep.py).
#
# Both arms sit under the preset's own 4096-token KV default (no override
# needed) -- CPU preflight (research/t3-88col-1024-nat/preflight.py)
# re-derives the ADMIT via the real harness gate before any GPU touch.
#
# Wired-knob: this window's projected peak is ~93 GiB class (well under the
# preset's own 96 GiB declared limit) -- no specific sysctl value is
# required for these small cells (unlike the 88e x 16k K-selection window's
# 112 GiB knob requirement). This script only RECORDS the live
# iogpu.wired_limit_mb for the receipt; it does not gate on a specific
# value.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_88col_1024_nat] wired-knob (recorded, not gated): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_88col_1024_nat] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_88col_1024_nat_wired_knob.txt" >&2

echo "[run_88col_1024_nat] CPU-only admission preflight (zero GPU touch) for both arms..." >&2
PYTHONPATH="$WT" "$PY" research/t3-88col-1024-nat/preflight.py \
  > "$WT/evals/tier2/t3_88col_1024_nat_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_88col_1024_nat_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_88col_1024_nat] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_88col_1024_nat_admission_preflight.log" >&2
  exit 2
fi
echo "[run_88col_1024_nat] preflight OK; both arms ADMIT at the preset's own default KV budget" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 3600 \
  -- \
  bash "$WT/research/t3-88col-1024-nat/run_88col_1024_nat_inner.sh"
