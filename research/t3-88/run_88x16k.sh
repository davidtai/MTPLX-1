#!/usr/bin/env bash
# OUTER launcher: T3 88 GiB envelope fix-candidate x16k matrix cell (2026-07-22).
#
# hy3-oq2e-rq4-88 (islands 79) REJECTs the config-time admission gate at
# EVERY swept KV level (constant 0.629 GiB over-book of its own 95 GiB
# limit -- evals/tier2/oq2e_rq4_88_16k.json, NOTES.md T4 entry). Two fix
# candidates are staged in benchmarks/presets.toml:
#   hy3-oq2e-rq4-88r ("r" = re-derived limit): islands 79 KEPT, memory-limit
#     re-derived to 96 GiB (harness fixed_bytes 95.629 GiB + 0.371 GiB
#     margin). David-approved 2026-07-22; this is the priority variant.
#   hy3-oq2e-rq4-88f ("f" = family-formula fallback): islands 78, memory-
#     limit kept at 95 GiB. Fallback only -- razor-thin 0.0034 GiB fits_fixed
#     margin (see research/envelope-admission-sweep-2026-07-22-88rf.json),
#     nowhere near the family's healthy 0.3-1.0 GiB band. Prefer 88r.
#
# Usage:
#   research/t3-88/run_88x16k.sh 88r     # or: 88f
#
# Runs AR+K1+K2+K3 (--hy3-depths 1,2,3; AR auto-runs as depth=0 regardless),
# 3 reps (3 independent process invocations -- retained_replicates is
# hardcoded 1 inside the harness, no --reps flag exists), 16k KV override.
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all is exported process-wide for
# every rep (see run_88x16k_inner.sh header for why -- same reasoning as
# research/t3-64-ab/run_window_inner.sh).
#
# CPU-only preflight (research/t3-88/preflight.py, single-cell gate mode)
# MUST pass before the guarded window opens -- if it REJECTs, this script
# exits 2 and NO GPU work happens, no qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a STANDALONE
# command (never chained, never shell `&`) -- box law: a chained window died
# to a hook timeout on 2026-07-21.
set -uo pipefail

VARIANT="${1:?usage: run_88x16k.sh <88r|88f>}"
case "$VARIANT" in
  88r|88f) ;;
  *) echo "[run_88x16k] unknown variant '$VARIANT' -- expected 88r or 88f" >&2; exit 9 ;;
esac
PRESET="hy3-oq2e-rq4-${VARIANT}"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_88x16k] CPU-only admission preflight for $PRESET @ 16384 KV (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-88/preflight.py --preset "$PRESET" --kv 16384 \
  > "$WT/evals/tier2/t3_88x16k_${VARIANT}_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_88x16k_${VARIANT}_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_88x16k] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_88x16k_${VARIANT}_admission_preflight.log" >&2
  exit 2
fi
echo "[run_88x16k] preflight OK; $PRESET ADMITs at 16384 KV" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 18000 \
  -- \
  bash "$WT/research/t3-88/run_88x16k_inner.sh" "$VARIANT"
