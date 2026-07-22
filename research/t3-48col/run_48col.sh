#!/usr/bin/env bash
# OUTER launcher: T3 48-envelope-column real-prefill matrix guarded window
# (2026-07-22). Matrix K locked = K1 (2e2ed99, #130 comment 5048661802).
#
# Preset `hy3-oq2e-rq4-48-cachelru` (ZERO islands, cache-heavy LRU -- David's
# <=64 GiB no-islands directive). Four arms in ONE guarded window (one model
# load per rep, one qwen-stop/flock hold covering all twelve process
# invocations): 48x1024, 48x16k, 48x64k, naturalistic-48. All real long-code
# prefill (except the naturalistic arm's short natural-language tail),
# decode 256, greedy, bf16 KV (--kv-quant omitted), K1 (--hy3-depths 1; AR
# row auto-runs alongside for free), 3 reps each.
#
# Memory-limit overrides (harness-derived by research/t3-48col/preflight.py,
# never hand-derived): NONE needed for this column -- every arm ADMITs at
# the preset's own declared 55 GiB limit. Its ~34 GiB elastic cache shrinks
# to absorb KV growth (33.98 -> 30.46 -> 15.23 GiB as kv_tokens goes
# 4096 -> 16640 -> 65792) while fixed_bytes grows in step (20.64 -> 24.47 ->
# 39.47 GiB), staying inside 55 GiB throughout (margins 0.3798 / 0.0668 /
# 0.2987 GiB).
#
# Sign-off (David's rule 7): every derived limit here (55 GiB) sits well
# under the sub-85/90 GiB sign-off band and the 112 GiB wired knob -- no
# fresh ask needed for this column.
#
# CPU-only preflight (research/t3-48col/preflight.py) MUST pass (all
# runnable arms ADMIT; any arm whose derived override would exceed the
# 112 GiB knob, or that still REJECTs at that override, is marked SKIP and
# excluded) before this guarded window opens -- if it fails, this script
# exits 2 and NO GPU work happens, no qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_48col] wired-knob (recorded, not gated at this margin): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_48col] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_48col_wired_knob.txt" >&2

echo "[run_48col] CPU-only admission preflight (zero GPU touch) for all four arms..." >&2
PYTHONPATH="$WT" "$PY" research/t3-48col/preflight.py \
  --output "$WT/evals/tier2/t3_48col_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_48col_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_48col] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_48col_admission_preflight.log" >&2
  exit 2
fi
echo "[run_48col] preflight OK; all runnable arms ADMIT" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-48col/run_48col_inner.sh"
