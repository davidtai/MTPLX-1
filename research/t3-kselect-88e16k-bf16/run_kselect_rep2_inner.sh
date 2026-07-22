#!/usr/bin/env bash
# INNER command for the T3 bf16-KV K-SELECTION window REP 2 recovery run
# (2026-07-22). Single rep only -- rep 1 already passed cleanly in the
# original window (see run_kselect_rep2.sh header for the root-cause
# writeup on why rep 2 needs a retry, not a full 3-rep re-run).
#
# The CALLER (run_kselect_rep2.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen.
#
# Same exact-lane env / preset / shape / memory-limit override as
# run_kselect_inner.sh's rep loop.
#
# Usage (never invoked directly -- always via run_kselect_rep2.sh):
#   run_kselect_rep2_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
MEMLIMIT=107189633024  # 99.8281 GiB; harness-derived, see preflight.py

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
  echo "[run_kselect_rep2_inner_bf16] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

rep=2
out="$OUTDIR/t3_kselect_88e16k_bf16_rep${rep}.json"
log="$OUTDIR/t3_kselect_88e16k_bf16_rep${rep}.log"
echo "[run_kselect_rep2_inner_bf16] === preset=$PRESET rep=$rep (RETRY) starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" \
  --contexts 16384 \
  --output-tokens 256 \
  --max-live-kv-tokens 16640 \
  --memory-limit "$MEMLIMIT" \
  --hy3-depths 1,2,3 \
  --output-json "$out" \
  >> "$log" 2>&1
rc=$?
echo "[run_kselect_rep2_inner_bf16] === preset=$PRESET rep=$rep (RETRY) exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
echo "[run_kselect_rep2_inner_bf16] preset=$PRESET rep=$rep (RETRY) exit=$rc $(date -u +%FT%TZ)" >&2

exit $rc
