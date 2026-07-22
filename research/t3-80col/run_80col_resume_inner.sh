#!/usr/bin/env bash
# INNER command for the T3 80-column RESUME guarded window (2026-07-22).
# The first 80col window was killed externally (driver-stop process-tree
# teardown) after completing: 1024 reps 1-3 and 16384 rep1 (all committed,
# 7331625). This resume runs ONLY the remaining reps:
#   ARM 2 (80x16k):        reps 2 3   (rep2 redone -- its first attempt lost
#                                      the K1 row to the kill)
#   ARM 3 (80x64k):        reps 1 2 3
#   ARM 4 (naturalistic):  reps 1 2 3
# Same preset/flags/overrides as run_80col_inner.sh (see its header).
# Never invoked directly -- always via run_80col_resume.sh.
set -uo pipefail

PRESET="hy3-oq2e-rq4-80"

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
  echo "[run_80col_resume_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

MEMLIMIT_16K=97525956608   # 90.8281 GiB; harness-derived, see preflight.py
MEMLIMIT_64K=113632083968  # 105.8281 GiB; harness-derived, see preflight.py

overall_rc=0

echo "[run_80col_resume_inner] === ARM 2 (80x16k) reps 2-3 starting $(date -u +%FT%TZ) ===" >&2
for rep in 2 3; do
  out="$OUTDIR/t3_80col_16384_rep${rep}.json"
  log="$OUTDIR/t3_80col_16384_rep${rep}.log"
  echo "[run_80col_resume_inner] === arm=16384 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 16384 \
    --output-tokens 256 \
    --max-live-kv-tokens 16640 \
    --memory-limit "$MEMLIMIT_16K" \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_80col_resume_inner] === arm=16384 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_80col_resume_inner] arm=16384 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_80col_resume_inner] === ARM 3 (80x64k) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_80col_65536_rep${rep}.json"
  log="$OUTDIR/t3_80col_65536_rep${rep}.log"
  echo "[run_80col_resume_inner] === arm=65536 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 65536 \
    --output-tokens 256 \
    --max-live-kv-tokens 65792 \
    --memory-limit "$MEMLIMIT_64K" \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_80col_resume_inner] === arm=65536 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_80col_resume_inner] arm=65536 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_80col_resume_inner] === ARM 4 (naturalistic ~265) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_80col_naturalistic_rep${rep}.json"
  log="$OUTDIR/t3_80col_naturalistic_rep${rep}.log"
  echo "[run_80col_resume_inner] === arm=naturalistic preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" research/t3-80col/run_naturalistic_80_rep.py \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_80col_resume_inner] === arm=naturalistic preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_80col_resume_inner] arm=naturalistic rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_80col_resume_inner] all remaining reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
