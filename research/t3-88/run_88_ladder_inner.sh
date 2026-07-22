#!/usr/bin/env bash
# INNER command for the T3 88 GiB fix-candidate K1-K7 depth-ladder window.
#
# The CALLER (run_88_ladder.sh -> run_with_qwen_stopped.py) already holds
# the exclusive MLX flock and has stopped qwen. ONE
# benchmark_q2_mtp_depth_matrix.py process, natural bench-shape (preset's
# own 4096-token max-live-kv-tokens default, no memory-limit override),
# --hy3-depths 1,2,3,4,5,6,7 (AR auto-runs as depth=0 regardless of what
# --hy3-depths requests).
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all, same convention as
# run_88x16k_inner.sh / research/t3-64-ab/run_window_inner.sh.
#
# Usage (never invoked directly -- always via run_88_ladder.sh):
#   run_88_ladder_inner.sh <88r|88f>
set -uo pipefail

VARIANT="${1:?usage: run_88_ladder_inner.sh <88r|88f>}"
PRESET="hy3-oq2e-rq4-${VARIANT}"

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
  echo "[run_88_ladder_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

out="$OUTDIR/t3_88_ladder_${VARIANT}.json"
log="$OUTDIR/t3_88_ladder_${VARIANT}.log"
echo "[run_88_ladder_inner] === preset=$PRESET K1-K7 ladder starting $(date -u +%FT%TZ) ===" | tee -a "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" --hy3-depths 1,2,3,4,5,6,7 \
  --output-json "$out" \
  > "$log" 2>&1
rc=$?
echo "[run_88_ladder_inner] === preset=$PRESET K1-K7 ladder exit=$rc $(date -u +%FT%TZ) ===" >&2
exit $rc
