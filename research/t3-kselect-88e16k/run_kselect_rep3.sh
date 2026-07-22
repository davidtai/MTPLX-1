#!/usr/bin/env bash
# OUTER launcher: T3 K-SELECTION window, REP 3 ONLY (recovery run, 2026-07-22).
#
# The original 3-rep window (run_kselect.sh) had its background wrapper
# process externally terminated (status: killed, NOT a box-law violation by
# the driver -- no kill/TaskStop/SIGSTOP was issued, no lock/qwen misuse)
# partway through rep 3 -- apparent implicit background-task duration cap
# in this sandbox (~65 min wall-clock; reps 1+2 each took ~22 min and
# completed+passed cleanly: evals/tier2/t3_kselect_88e16k_kv4_rep1.json and
# _rep2.json both status=passed, 4/4 observations, before the cutoff).
# Post-kill verification (read-only, no destructive action taken): qwen
# 127.0.0.1:8080 healthy (200, model list responds), /tmp/mtplx-gpu-
# exclusive.lock present but held by nobody (lsof empty), no benchmark_q2_
# mtp_depth_matrix.py process alive. This matches the box law's documented
# recovery precedent (verify qwen healthy + lock clean after a foreign
# teardown, then retry) -- retrying ONLY the missing rep (3), not re-running
# 1+2, to keep this window comfortably under the observed ~65 min cap
# (single rep ~22 min).
#
# Same preset/shape as the original window, unchanged: hy3-oq2e-rq4-88e,
# --contexts 16384 --output-tokens 256 --max-live-kv-tokens 16640
# --kv-quant q4 --hy3-depths 1,2,3, greedy (harness default), no
# --memory-limit override (preset's own 96GiB limit ADMITs this shape
# directly). CPU-only admission preflight re-run below per box law ("EVERY
# config through the harness before GPU") even though nothing changed.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter).
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
KV=16640

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_kselect_rep3] CPU-only admission preflight for $PRESET @ kv=$KV kv_quant=q4 (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-kselect-88e16k/preflight.py --kv "$KV" --kv-quant q4 \
  > "$WT/evals/tier2/kselect_88e16k_rep3_admission_preflight.json" \
  2> "$WT/evals/tier2/kselect_88e16k_rep3_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_kselect_rep3] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/kselect_88e16k_rep3_admission_preflight.log" >&2
  exit 2
fi
echo "[run_kselect_rep3] preflight OK; $PRESET ADMITs at kv=$KV kv_quant=q4" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 3600 \
  -- \
  bash "$WT/research/t3-kselect-88e16k/run_kselect_rep3_inner.sh"
