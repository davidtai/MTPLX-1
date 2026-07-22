#!/usr/bin/env bash
# INNER command for the T3 K-SELECTION window REP 3 recovery run
# (2026-07-22). Single rep only -- reps 1+2 already passed cleanly in the
# original window (see run_kselect_rep3.sh header for why this is a
# rep-3-only retry, not a full 3-rep re-run).
#
# The CALLER (run_kselect_rep3.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen.
#
# Same exact-lane env / preset / shape as run_kselect_inner.sh's rep loop.
#
# Usage (never invoked directly -- always via run_kselect_rep3.sh):
#   run_kselect_rep3_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
"
RC=$?
if [[ "$RC" != 0 ]]; then
  echo "[run_kselect_rep3_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

rep=3
out="$OUTDIR/t3_kselect_88e16k_kv4_rep${rep}.json"
log="$OUTDIR/t3_kselect_88e16k_kv4_rep${rep}.log"
echo "[run_kselect_rep3_inner] === preset=$PRESET rep=$rep (RETRY) starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" \
  --contexts 16384 \
  --output-tokens 256 \
  --max-live-kv-tokens 16640 \
  --kv-quant q4 \
  --hy3-depths 1,2,3 \
  --output-json "$out" \
  >> "$log" 2>&1
rc=$?
echo "[run_kselect_rep3_inner] === preset=$PRESET rep=$rep (RETRY) exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
echo "[run_kselect_rep3_inner] preset=$PRESET rep=$rep (RETRY) exit=$rc $(date -u +%FT%TZ)" >&2

exit $rc
