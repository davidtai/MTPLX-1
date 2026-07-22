#!/usr/bin/env bash
# INNER command for CONTINUATION window 2: Arm B (cache-policy lru), 3 reps.
#
# Window 1 (run_window.sh/run_window_inner.sh) completed Arm A (3/3 reps)
# and Arm B frequency (3/3 reps) cleanly, then was killed by the harness
# (background-task lifetime, ~60 min observed -- not a box-law violation,
# not a foreign-lock issue: qwen was restored and the flock was released
# cleanly by run_with_qwen_stopped.py's finally-block before this was
# noticed) right as armB_lru rep=1 started loading. That rep's output file
# was a setup-phase-only checkpoint (no measurement) and was deleted before
# this window. Splitting the remainder into two shorter windows (this one,
# then run_window3_patch_inner.sh) so each comfortably finishes well inside
# whatever the observed ~60 min ceiling turns out to be.
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
  echo "[run_window2] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

ARM_A_MEMLIMIT=80262201344    # 74.75 GiB: 71 GiB preset + (16384-4096)*327680 KV delta
ARM_B_CACHE_LIMIT=53678702592 # 49.9921875 GiB, derived by research/t3-64-ab/preflight.py

overall_rc=0

for rep in 1 2 3; do
  out="$OUTDIR/t3_64x16k_armB_lru_rep${rep}.json"
  log="$OUTDIR/t3_64x16k_armB_lru_rep${rep}.log"
  echo "[run_window2] === lane=armB_lru rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --model hy3-oq2e --contexts 1024 --output-tokens 1024 \
    --memory-limit "$ARM_A_MEMLIMIT" --runtime-reserve 7GiB \
    --expert-cache-limit "$ARM_B_CACHE_LIMIT" --max-live-kv-tokens 16384 \
    --cache-policy lru --cache-scope layer --slot-layout component-banks \
    --hy3-router-kernel mpp-fp32-splitk-r1-fused-r2 --verify-strategy batched \
    --expert-integrity headers-only --proj-requant q4 \
    --split-route-release deferred --hy3-depths 1,2,3 \
    --output-json "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[run_window2] === lane=armB_lru rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_window2] armB_lru reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
