#!/usr/bin/env bash
# INNER command for CONTINUATION window 3: the K3-exact patch lane, 3 reps.
# Arm A's preset config unchanged, --max-live-kv-tokens 32768, the SAME
# 79.75 GiB override research/t3-64x32k already GPU-verified as ADMIT, K3
# only (--hy3-depths 3; AR auto-runs as d0 regardless, giving the reference
# row the K3 token-hash comparison needs).
set -uo pipefail

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
  echo "[run_window3] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

PATCH_MEMLIMIT=85630910464  # 79.75 GiB: same override research/t3-64x32k already GPU-verified

overall_rc=0

for rep in 1 2 3; do
  out="$OUTDIR/t3_64x32k_k3exact_rep${rep}.json"
  log="$OUTDIR/t3_64x32k_k3exact_rep${rep}.log"
  echo "[run_window3] === lane=patch_k3exact rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset hy3-oq2e-rq4-64 --max-live-kv-tokens 32768 \
    --memory-limit "$PATCH_MEMLIMIT" --hy3-depths 3 \
    --output-json "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[run_window3] === lane=patch_k3exact rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_window3] patch_k3exact reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
