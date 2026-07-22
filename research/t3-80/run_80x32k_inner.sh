#!/usr/bin/env bash
# INNER command for the T3 80 GiB envelope matrix x32k guarded window.
#
# The CALLER (run_80x32k.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen. Runs 3 reps SERIALLY (one model
# resident at a time, per box law); each rep is its own
# benchmark_q2_mtp_depth_matrix.py process (memory fully releases between
# reps). --hy3-depths 1,2,3 requests K1/K2/K3; AR (depth=0) auto-runs
# regardless of what --hy3-depths requests, so every rep still yields an AR
# reference row.
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all is exported globally for
# every rep in this window -- same convention as research/t3-80/
# run_80x16k_inner.sh / research/t3-88/run_88x16k_inner.sh.
#
# This cell is PARTIAL residency (islands 69 of ~79 total expert layers --
# 10 streamed), unlike the -88r cell's full residency: expect real streaming
# telemetry (expert_requests > 0, hit_rate < 1), not the loads=0 full-
# residency signature.
#
# Usage (never invoked directly -- always via run_80x32k.sh):
#   run_80x32k_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-80"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

# Cross-check mtplx resolves to THIS worktree, not the stale parent checkout
# (the editable-install finder trap documented in evals/tier2/NOTES.md's T3
# 88x16k ops notes).
"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
"
RC=$?
if [[ "$RC" != 0 ]]; then
  echo "[run_80x32k_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

# Exact memory-limit override for 32768 KV, validated ADMIT by
# research/t3-80/preflight.py (CPU-only gate, zero GPU touch) and recorded
# in research/envelope-admission-sweep-2026-07-22-80.json:
#   87 GiB preset base + (32768-4096)*327680 KV delta = 95.75 GiB
#   = 102,810,779,648 B (fixed_bytes 94.8872 GiB, implied admission total
#   [fixed+cache] 95.7277 GiB -- the "95.73 GiB" sweep-verified figure).
MEMLIMIT=102810779648  # 95.75 GiB

overall_rc=0
for rep in 1 2 3; do
  out="$OUTDIR/t3_80x32k_bf16_rep${rep}.json"
  log="$OUTDIR/t3_80x32k_bf16_rep${rep}.log"
  echo "[run_80x32k_inner] === preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" --max-live-kv-tokens 32768 \
    --memory-limit "$MEMLIMIT" --hy3-depths 1,2,3 \
    --output-json "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[run_80x32k_inner] === preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_80x32k_inner] all reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
