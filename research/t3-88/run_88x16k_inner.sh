#!/usr/bin/env bash
# INNER command for the T3 88 GiB fix-candidate x16k guarded window.
#
# The CALLER (run_88x16k.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen. Runs 3 reps SERIALLY (one model
# resident at a time, per box law); each rep is its own
# benchmark_q2_mtp_depth_matrix.py process (memory fully releases between
# reps). --hy3-depths 1,2,3 requests K1/K2/K3; AR (depth=0) auto-runs
# regardless of what --hy3-depths requests, so every rep still yields an AR
# reference row.
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all is exported globally for
# every rep in this window -- same convention as research/t3-64-ab/
# run_window_inner.sh (confirmed from receipts: the router kernel is
# configured once at model load inside mtplx/runtime.py's _load_impl,
# os.environ.get("MTPLX_HY3_ROUTER_SPLITK_M1", "mtp"), read a single time
# per process; every champion receipt that carries it shows it applied
# process-wide, affecting the AR lane's own parity too, not just the
# deepest K).
#
# Usage (never invoked directly -- always via run_88x16k.sh):
#   run_88x16k_inner.sh <88r|88f>
set -uo pipefail

VARIANT="${1:?usage: run_88x16k_inner.sh <88r|88f>}"
PRESET="hy3-oq2e-rq4-${VARIANT}"

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
  echo "[run_88x16k_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# apply_preset_defaults() only os.environ.setdefault()s these from the
# preset's own [preset.NAME.env] table anyway (both 88r/88f inherit them
# from hy3-oq2e-rq4-88 via `extends`), so exporting here is belt-and-braces,
# not strictly required -- kept for parity with the t3-64-ab convention.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

# Exact memory-limit overrides for 16384 KV, validated ADMIT by
# research/t3-88/preflight.py (CPU-only gate, zero GPU touch) and recorded
# in research/envelope-admission-sweep-2026-07-22-88rf.json:
#   88r: 96 GiB preset + (16384-4096)*327680 KV delta = 99.75 GiB = 107105746944 B
#   88f: 95 GiB preset + (16384-4096)*327680 KV delta = 98.75 GiB = 106032005120 B
case "$VARIANT" in
  88r) MEMLIMIT=107105746944 ;;  # 99.75 GiB
  88f) MEMLIMIT=106032005120 ;;  # 98.75 GiB
  *) echo "[run_88x16k_inner] unknown variant '$VARIANT'" >&2; exit 9 ;;
esac

overall_rc=0
for rep in 1 2 3; do
  out="$OUTDIR/t3_88x16k_bf16_${VARIANT}_rep${rep}.json"
  log="$OUTDIR/t3_88x16k_bf16_${VARIANT}_rep${rep}.log"
  echo "[run_88x16k_inner] === preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" --max-live-kv-tokens 16384 \
    --memory-limit "$MEMLIMIT" --hy3-depths 1,2,3 \
    --output-json "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[run_88x16k_inner] === preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_88x16k_inner] all reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
