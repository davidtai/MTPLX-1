#!/usr/bin/env bash
# OUTER launcher: T3 80-envelope-column real-prefill matrix guarded window
# (2026-07-22). Matrix K locked = K1 (2e2ed99, #130 comment 5048661802).
#
# Preset `hy3-oq2e-rq4-80` (islands 69, used AS-IS -- David's no-islands-
# change directive applies only to <=64 GiB presets). Four arms in ONE
# guarded window (one model load per rep, one qwen-stop/flock hold covering
# all twelve process invocations): 80x1024, 80x16k, 80x64k, naturalistic-80.
# All real long-code prefill (except the naturalistic arm's short natural-
# language tail), decode 256, greedy, bf16 KV (--kv-quant omitted), K1
# (--hy3-depths 1; AR row auto-runs alongside for free), 3 reps each.
#
# Memory-limit overrides (harness-derived by research/t3-80col/preflight.py,
# never hand-derived):
#   1024 / naturalistic arms: no override (preset's own 87 GiB declared
#     limit ADMITs at the preset's own 4096-token KV default; margin
#     +0.0223 GiB).
#   16384 arm: --max-live-kv-tokens 16640, --memory-limit 97525956608
#     (90.8281 GiB; fixed=89.9654 GiB, margin +0.0223 GiB).
#   65536 arm: --max-live-kv-tokens 65792, --memory-limit 113632083968
#     (105.8281 GiB; fixed=104.9654 GiB, margin +0.0223 GiB) -- well under
#     the 112 GiB (114688 MiB) wired knob; this script only RECORDS the
#     live iogpu.wired_limit_mb for the receipt, it does not require a
#     specific value (unlike the 88e x 16k K-selection window, which
#     REQUIRES 114688 because its own override sits closer to the knob).
#
# Sign-off (David's rule 7): >85 GiB projected peaks are pre-granted for
# this column (handoff rule 7 + the 80x64k live grant, HANDOFF-hy3-matrix-
# restart.md) -- not asking again.
#
# CPU-only preflight (research/t3-80col/preflight.py) MUST pass (all
# runnable arms ADMIT; any arm whose derived override would exceed the
# 112 GiB knob is marked SKIP and excluded) before this guarded window
# opens -- if it fails, this script exits 2 and NO GPU work happens, no
# qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_80col] wired-knob (recorded, not gated at this margin): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_80col] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_80col_wired_knob.txt" >&2

echo "[run_80col] CPU-only admission preflight (zero GPU touch) for all four arms..." >&2
PYTHONPATH="$WT" "$PY" research/t3-80col/preflight.py \
  --output "$WT/evals/tier2/t3_80col_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_80col_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_80col] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_80col_admission_preflight.log" >&2
  exit 2
fi
echo "[run_80col] preflight OK; all runnable arms ADMIT" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-80col/run_80col_inner.sh"
