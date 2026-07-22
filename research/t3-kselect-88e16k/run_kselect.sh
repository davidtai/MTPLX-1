#!/usr/bin/env bash
# OUTER launcher: T3 K-SELECTION guarded window (2026-07-22).
#
# Mission (HANDOFF-hy3-matrix-restart.md work-queue item 3, David-confirmed
# shape): this is THE cell that decides which K locks matrix-wide. Preset
# `hy3-oq2e-rq4-88e` (efficient-88, committed c513c6d), REAL prefill
# (--contexts 16384), decode 256, kv4 serving config
# (--max-live-kv-tokens 16640, --kv-quant q4), greedy (harness default),
# AR + K1/K2/K3 (--hy3-depths 1,2,3 -- AR/depth=0 auto-runs regardless),
# 3 reps (3 independent process invocations -- retained_replicates is
# hardcoded 1 inside the harness, no --reps flag exists; same convention as
# research/t3-88/run_88x16k_inner.sh).
#
# hy3-oq2e-rq4-88e already declares memory-limit=96GiB (inherited from
# -88r) -- NO --memory-limit override here, unlike the bf16 t3-88 bench-
# shape cells (those overrode the limit to absorb a bf16 KV-delta; this
# cell runs under kv4, whose delta is already inside the preset's own
# admission math -- see preflight.py, ADMIT at 96 GiB with +0.19 GiB
# margin, no override needed).
#
# Sign-off (David's rule 7): projected peak ~95.8 GiB is >85 GiB but is
# covered by the standing "88x16k precedent grant" in the handoff
# (rule 7 grants list + work-queue item 1 note) -- cited here, not asking
# again.
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all is exported inside
# run_kselect_inner.sh (in NO preset, must be set manually, read once per
# process -- see run_kselect_inner.sh header, same convention as t3-80/
# t3-88/t3-pilot-realprefill).
#
# CPU-only preflight (research/t3-kselect-88e16k/preflight.py) MUST pass
# before the guarded window opens -- if it REJECTs, this script exits 2 and
# NO GPU work happens, no qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
KV=16640

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_kselect] CPU-only admission preflight for $PRESET @ kv=$KV kv_quant=q4 (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-kselect-88e16k/preflight.py --kv "$KV" --kv-quant q4 \
  > "$WT/evals/tier2/kselect_88e16k_admission_preflight.json" \
  2> "$WT/evals/tier2/kselect_88e16k_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_kselect] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/kselect_88e16k_admission_preflight.log" >&2
  exit 2
fi
echo "[run_kselect] preflight OK; $PRESET ADMITs at kv=$KV kv_quant=q4" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 28800 \
  -- \
  bash "$WT/research/t3-kselect-88e16k/run_kselect_inner.sh"
