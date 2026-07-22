#!/usr/bin/env bash
# INNER command for the T3 88-column completion guarded window (2026-07-22).
#
# The CALLER (run_88col_1024_nat.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen. Runs BOTH remaining
# 88-envelope arms serially inside ONE window (one model load per process,
# six processes total: 3 reps arm1 + 3 reps arm2), per box law (one window
# at a time, one model resident at a time -- memory fully releases between
# processes).
#
# ARM 1: fresh matrix-shape cell, preset hy3-oq2e-rq4-88e, --contexts 1024
# --output-tokens 256 --hy3-depths 1 (AR row auto-runs alongside), bf16 KV
# (--kv-quant omitted), max-live-kv-tokens left at the preset's own argparse
# default (4096; 1024+256=1280 tokens sit comfortably inside it -- same
# byte-budget already ADMITted at the preset's own control point, re-derived
# by research/t3-88col-1024-nat/preflight.py). NOTE (per launch instructions):
# this is a FRESH matrix-shape cell, distinct from the old bench-lane
# 1024/1024 receipts (a different shape) -- not reused/compared beyond a
# label.
#
# ARM 2: naturalistic short-prompt arm. hy3-oq2e has no --prompt-tail CLI
# flag, so this calls research/t3-88col-1024-nat/run_naturalistic_rep.py,
# which imports benchmark_q2_mtp_depth_matrix directly and monkeypatches
# _requests_from_args to inject research/naturalistic_prompt_v1.txt as
# prompt_tail_text for the hy3-oq2e request (see that script's header for
# the full context-token derivation: 265 = 264-token chat-templated tail +
# 1 structurally-required filler token; run_depth_matrix's own hard gate on
# prompt_release_valid/prompt_tail_preserved rejects the naive exact-264
# reading, direct-probed offline before this window). Same preset, same K1
# (+free AR), same bf16 KV, same unmodified 4096 KV-token default.
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all -- process-wide, all reps,
# both arms, same convention as every other t3-* window.
#
# Usage (never invoked directly -- always via run_88col_1024_nat.sh):
#   run_88col_1024_nat_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

# Cross-check mtplx resolves to THIS worktree, not a stale parent checkout
# (the editable-install finder trap documented in evals/tier2/NOTES.md).
"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
"
RC=$?
if [[ "$RC" != 0 ]]; then
  echo "[run_88col_1024_nat_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps, both arms.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# apply_preset_defaults() only os.environ.setdefault()s these from the
# preset's own [preset.hy3-oq2e-rq4-88e.env] table anyway; kept explicit for
# parity with every other t3-* window.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

overall_rc=0

echo "[run_88col_1024_nat_inner] === ARM 1 (fresh 1024 cell) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_88col_1024_rep${rep}.json"
  log="$OUTDIR/t3_88col_1024_rep${rep}.log"
  echo "[run_88col_1024_nat_inner] === arm1 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 1024 \
    --output-tokens 256 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_88col_1024_nat_inner] === arm1 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_88col_1024_nat_inner] arm1 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_88col_1024_nat_inner] === ARM 2 (naturalistic ~265 cell) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_88col_naturalistic_rep${rep}.json"
  log="$OUTDIR/t3_88col_naturalistic_rep${rep}.log"
  echo "[run_88col_1024_nat_inner] === arm2 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" research/t3-88col-1024-nat/run_naturalistic_rep.py \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_88col_1024_nat_inner] === arm2 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_88col_1024_nat_inner] arm2 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_88col_1024_nat_inner] all reps attempted (both arms); overall_rc=$overall_rc" >&2
exit $overall_rc
