#!/usr/bin/env bash
# INNER command for the T3 64 GiB island-vs-cache A/B guarded window.
#
# The CALLER (run_window.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen. Runs every lane SERIALLY (one
# model resident at a time, per box law): armA, then armB (frequency), then
# armB (lru), then the K3-exact patch lane. Each rep is its own
# benchmark_q2_mtp_depth_matrix.py process (memory fully releases between
# reps/lanes). 3 reps per lane -- retained_replicates is hardcoded 1 inside
# the harness (no --reps flag exists), so "3 reps" here means 3 independent
# process invocations per lane, aggregated after the window by
# research/t3-64-ab/aggregate.py (CPU-only, no GPU touch).
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all is exported globally for
# EVERY lane in this window. Confirmed from receipts (evals/tier2/NOTES.md,
# "PARITY-STAMPED CHAMPION" / "Router M1 scope A/B"): the router kernel is
# configured once at model load inside mtplx/runtime.py's _load_impl
# (os.environ.get("MTPLX_HY3_ROUTER_SPLITK_M1", "mtp"), read a single time
# per process), and benchmark_q2_mtp_depth_matrix.py loads the model ONCE
# per process and reuses it for every requested depth (AR/K1/K2/K3 all share
# one load_model call) -- there is no mechanism to scope this env var to a
# single K value within one invocation. Every receipt that carries it
# (champion-41 reproduction, PARITY-STAMPED CHAMPION: AR 40.18 parity True /
# K2 42.18 parity True) shows it applied process-wide, affecting the AR
# lane's own parity too, not just the deepest K. So it is set here for the
# whole window, all lanes, all depths -- not "K3 only".
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

# Cross-check mtplx resolves to THIS worktree, not the stale parent checkout
# (the editable-install finder trap documented in evals/tier2/NOTES.md's T3
# 88x16k ops notes -- run_speed_arms_inner.sh hit the same class of failure).
"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
"
RC=$?
if [[ "$RC" != 0 ]]; then
  echo "[run_window_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all lanes, all K values.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# Replicated manually because Arm B does not go through --preset (it cannot
# -- see preflight.py's docstring on why zero islands requires bypassing the
# preset machinery). apply_preset_defaults() only os.environ.setdefault()s
# these for Arm A/patch anyway, so exporting them here is a no-op for those
# two lanes and required for Arm B; identical env for all four lane configs.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

ARM_A_MEMLIMIT=80262201344    # 74.75 GiB: 71 GiB preset + (16384-4096)*327680 KV delta
ARM_B_CACHE_LIMIT=53678702592 # 49.9921875 GiB, derived by research/t3-64-ab/preflight.py
PATCH_MEMLIMIT=85630910464    # 79.75 GiB: same override research/t3-64x32k already GPU-verified

overall_rc=0

run_rep() {
  # $1=lane_label $2=rep_index(1-3) $3...=extra benchmark_q2_mtp_depth_matrix.py args
  local label="$1"; local rep="$2"; shift 2
  local out="$OUTDIR/t3_64x16k_${label}_rep${rep}.json"
  local log="$OUTDIR/t3_64x16k_${label}_rep${rep}.log"
  echo "[run_window_inner] === lane=$label rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    "$@" \
    --output-json "$out" \
    > "$log" 2>&1
  local rc=$?
  echo "[run_window_inner] === lane=$label rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  return $rc
}

# --- Part 1: Arm A (islands 52), 16k KV, AR+K1+K2+K3, 3 reps ---
for rep in 1 2 3; do
  run_rep armA "$rep" \
    --preset hy3-oq2e-rq4-64 --max-live-kv-tokens 16384 \
    --memory-limit "$ARM_A_MEMLIMIT" --hy3-depths 1,2,3
  rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc
done

# --- Part 2a: Arm B, cache-policy frequency, 0 islands, 16k KV, 3 reps ---
for rep in 1 2 3; do
  run_rep armB_frequency "$rep" \
    --model hy3-oq2e --contexts 1024 --output-tokens 1024 \
    --memory-limit "$ARM_A_MEMLIMIT" --runtime-reserve 7GiB \
    --expert-cache-limit "$ARM_B_CACHE_LIMIT" --max-live-kv-tokens 16384 \
    --cache-policy frequency --cache-scope layer --slot-layout component-banks \
    --hy3-router-kernel mpp-fp32-splitk-r1-fused-r2 --verify-strategy batched \
    --expert-integrity headers-only --proj-requant q4 \
    --split-route-release deferred --hy3-depths 1,2,3
  rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc
done

# --- Part 2b: Arm B, cache-policy lru, 0 islands, 16k KV, 3 reps ---
for rep in 1 2 3; do
  run_rep armB_lru "$rep" \
    --model hy3-oq2e --contexts 1024 --output-tokens 1024 \
    --memory-limit "$ARM_A_MEMLIMIT" --runtime-reserve 7GiB \
    --expert-cache-limit "$ARM_B_CACHE_LIMIT" --max-live-kv-tokens 16384 \
    --cache-policy lru --cache-scope layer --slot-layout component-banks \
    --hy3-router-kernel mpp-fp32-splitk-r1-fused-r2 --verify-strategy batched \
    --expert-integrity headers-only --proj-requant q4 \
    --split-route-release deferred --hy3-depths 1,2,3
  rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc
done

# --- Part 3: Patch lane -- Arm A's config, 32k KV, K3 ONLY, 3 reps ---
for rep in 1 2 3; do
  local_out="$OUTDIR/t3_64x32k_k3exact_rep${rep}.json"
  local_log="$OUTDIR/t3_64x32k_k3exact_rep${rep}.log"
  echo "[run_window_inner] === lane=patch_k3exact rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$local_log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset hy3-oq2e-rq4-64 --max-live-kv-tokens 32768 \
    --memory-limit "$PATCH_MEMLIMIT" --hy3-depths 3 \
    --output-json "$local_out" \
    > "$local_log" 2>&1
  rc=$?
  echo "[run_window_inner] === lane=patch_k3exact rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_window_inner] all lanes/reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
