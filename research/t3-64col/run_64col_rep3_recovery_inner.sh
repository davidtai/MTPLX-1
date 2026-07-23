#!/usr/bin/env bash
# INNER: 64x64k rep3 single-rep recovery. Same flags as run_64col_inner.sh
# ARM 3 (see its header); overwrites the partial rep3 receipt.
set -uo pipefail

PRESET="hy3-oq2e-rq4-64-cachelru"
WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
" || exit 9

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

out="$OUTDIR/t3_64col_65536_rep3.json"
log="$OUTDIR/t3_64col_65536_rep3.log"
echo "[64col-rep3-recovery] === arm=65536 preset=$PRESET rep=3 (recovery) starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" \
  --contexts 65536 \
  --output-tokens 256 \
  --max-live-kv-tokens 65792 \
  --hy3-depths 1 \
  --output-json "$out" \
  >> "$log" 2>&1
rc=$?
echo "[64col-rep3-recovery] === arm=65536 rep=3 exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
echo "[64col-rep3-recovery] exit=$rc" >&2
exit $rc
