#!/usr/bin/env bash
# INNER command for the T3-pre 64x32k kv-mode speed window.
#
# The CALLER (run_speed_arms.sh -> run_with_qwen_stopped.py) already holds
# the exclusive MLX flock and has stopped qwen. Runs the three paired arms
# SERIALLY (one model resident at a time, per box law): bf16 KV control
# first (also the T3 matrix-cell receipt for 64x32k bf16 KV), then kv8, then
# kv4. Each arm is its own benchmark_q2_mtp_depth_matrix.py process (memory
# fully releases between arms). Same --preset, same --max-live-kv-tokens,
# same --memory-limit override (79.75 GiB, a shared ceiling per the
# comparability rule -- NOT sized per-arm); only --kv-quant differs.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

# Cross-check mtplx resolves to THIS worktree, not the stale parent checkout
# (the editable-install finder trap documented in evals/litellm_hy3/handler.py
# and evals/tier2/NOTES.md's T3 88x16k ops notes: sys.path[0] defaults to cwd
# for `-c`/relative invocations, and the parent repo root also ships an
# mtplx/ package that wins if cwd is wrong).
"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
"
RC=$?
if [[ "$RC" != 0 ]]; then
  echo "[run_speed_arms_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

MEMLIMIT_BYTES=85630910464   # 79.75 GiB: 71 GiB preset + (32768-4096)*327680 KV delta

run_arm() {
  local label="$1"; shift
  local out="$OUTDIR/t3_64x32k_${label}.json"
  local log="$OUTDIR/t3_64x32k_${label}.log"
  echo "[run_speed_arms_inner] === arm=$label starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset hy3-oq2e-rq4-64 \
    --max-live-kv-tokens 32768 \
    --memory-limit "$MEMLIMIT_BYTES" \
    --hy3-depths 1,2,3 \
    "$@" \
    --output-json "$out" \
    > "$log" 2>&1
  local rc=$?
  echo "[run_speed_arms_inner] === arm=$label exit=$rc $(date -u +%FT%TZ) ===" >&2
  return $rc
}

overall_rc=0

run_arm bf16
rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc

run_arm kv8 --kv-quant q8
rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc

run_arm kv4 --kv-quant q4
rc=$?; [[ $rc -eq 0 ]] || overall_rc=$rc

echo "[run_speed_arms_inner] all arms attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
