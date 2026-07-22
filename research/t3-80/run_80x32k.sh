#!/usr/bin/env bash
# OUTER launcher: T3 80 GiB envelope matrix cell, 32768 max-live-kv-tokens,
# bf16 KV (2026-07-22). David-signed-off >85 GiB cell (granted live
# 2026-07-22 -- see evals/tier2/NOTES.md entry for this campaign).
#
# Preset `hy3-oq2e-rq4-80` (islands 69) used AS-IS -- David's no-islands-
# change directive applies only to <=64 GiB presets, so this is unmodified.
#
# Runs AR+K1+K2+K3 (--hy3-depths 1,2,3; AR auto-runs as depth=0 regardless),
# 3 reps, 32768 KV override. exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all
# exported process-wide for every rep -- same convention as
# research/t3-80/run_80x16k_inner.sh / research/t3-88/run_88x16k_inner.sh.
#
# Memory-limit override for this cell: preset's own 87 GiB base +
# (32768-4096)*327,680 B KV delta = 102,810,779,648 B = 95.75 GiB exactly --
# CPU-verified via research/t3-80/preflight.py (fixed_bytes 94.8872 GiB,
# implied admission total [fixed+cache] 95.7277 GiB, i.e. the "95.73 GiB"
# sweep-verified admission figure) prior to this script ever existing.
#
# CPU-only preflight (research/t3-80/preflight.py, single-cell gate mode)
# MUST pass before the guarded window opens -- if it REJECTs, this script
# exits 2 and NO GPU work happens, no qwen stop, no flock acquired.
#
# This window launches only AFTER run_80x16k.sh's wrapper has fully exited
# (window 1 -> window 2, strictly serial; box law: one model resident at a
# time, only one guarded window open at once).
#
# Launch this via the Bash tool with run_in_background:true, as a STANDALONE
# command (never chained, never shell `&`) -- box law: a chained window died
# to a hook timeout on 2026-07-21.
set -uo pipefail

PRESET="hy3-oq2e-rq4-80"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_80x32k] CPU-only admission preflight for $PRESET @ 32768 KV (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-80/preflight.py --kv 32768 \
  > "$WT/evals/tier2/t3_80x32k_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_80x32k_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_80x32k] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_80x32k_admission_preflight.log" >&2
  exit 2
fi
echo "[run_80x32k] preflight OK; $PRESET ADMITs at 32768 KV" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 18000 \
  -- \
  bash "$WT/research/t3-80/run_80x32k_inner.sh"
