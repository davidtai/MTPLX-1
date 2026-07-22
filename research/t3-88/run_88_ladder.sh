#!/usr/bin/env bash
# OUTER launcher: T3 88 GiB fix-candidate K1-K7 depth ladder (2026-07-22).
#
# Variant of run_88x16k.sh's mission for the depth-scaling question instead
# of the KV-scaling question: one guarded window, natural bench-shape (the
# preset's own 4096-token max-live-kv-tokens control default -- no KV
# override needed, since both hy3-oq2e-rq4-88r and hy3-oq2e-rq4-88f ADMIT at
# their own declared memory-limit with no override applied, per
# research/envelope-admission-sweep-2026-07-22-88rf.json), full K1..K7 depth
# sweep (--hy3-depths 1,2,3,4,5,6,7; AR auto-runs as depth=0 regardless).
# ONE invocation covers every depth in a single model load (the harness
# loads once per process and reuses it for every requested depth), so this
# is a single rep, not the 3-rep convention run_88x16k.sh uses for its
# timing-precision receipt -- this ladder is exploratory (does acceptance /
# tok-s degrade past K3?), not a championship-precision measurement.
#
# Usage:
#   research/t3-88/run_88_ladder.sh 88r     # or: 88f
#
# CPU-only preflight (research/t3-88/preflight.py, single-cell gate mode,
# --kv 4096 -- the preset's own control default) MUST pass before the
# guarded window opens -- if it REJECTs, this script exits 2 and NO GPU work
# happens, no qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a STANDALONE
# command (never chained, never shell `&`) -- box law: a chained window died
# to a hook timeout on 2026-07-21.
set -uo pipefail

VARIANT="${1:?usage: run_88_ladder.sh <88r|88f>}"
case "$VARIANT" in
  88r|88f) ;;
  *) echo "[run_88_ladder] unknown variant '$VARIANT' -- expected 88r or 88f" >&2; exit 9 ;;
esac
PRESET="hy3-oq2e-rq4-${VARIANT}"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_88_ladder] CPU-only admission preflight for $PRESET @ 4096 KV (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-88/preflight.py --preset "$PRESET" --kv 4096 \
  > "$WT/evals/tier2/t3_88_ladder_${VARIANT}_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_88_ladder_${VARIANT}_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_88_ladder] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_88_ladder_${VARIANT}_admission_preflight.log" >&2
  exit 2
fi
echo "[run_88_ladder] preflight OK; $PRESET ADMITs at its own 4096 KV default" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 18000 \
  -- \
  bash "$WT/research/t3-88/run_88_ladder_inner.sh" "$VARIANT"
