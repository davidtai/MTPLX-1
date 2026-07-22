#!/usr/bin/env bash
# INNER command for the T3 real-prefill PILOT guarded window (2026-07-22).
# Mechanics validation, NOT a cell of record: ONE 1-rep run of an actual
# 16384-token prefill (--contexts 16384, prompt_style="coding-agent",
# hard-asserted len(prompt_ids)==16384 -- see scripts/
# benchmark_q2_mtp_depth_matrix.py:2961-3005) through the zero-island
# cache-heavy-LRU 32 GiB envelope preset under kv4 KV.
#
# The CALLER (run_pilot.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen.
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all is exported here -- this
# env is in NO preset (HANDOFF-hy3-matrix-restart.md "MECHANICS SOLVED"),
# read once per process at model load inside mtplx/runtime.py's
# _load_impl, applies uniformly to every K arm run in this process
# (only K3 requested here via --hy3-depths 3, but AR auto-runs regardless).
#
# CPU admission preflight already passed before this window opened (see
# run_pilot.sh) -- ADMIT margin +0.2147 GiB at kv=16640/kv_quant=q4,
# preset's own 39 GiB memory-limit used AS-IS, no --memory-limit override.
#
# Usage (never invoked directly -- always via run_pilot.sh):
#   run_pilot_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-32-cachelru"

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
  echo "[run_pilot_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, this pilot's one rep.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# apply_preset_defaults() only os.environ.setdefault()s these from the
# preset's own [preset.hy3-oq2e-rq4-32-cachelru.env] table anyway, so
# exporting here is belt-and-braces, not strictly required -- kept for
# parity with the t3-80 / t3-88 convention.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

out="$OUTDIR/pilot_realprefill_32cachelru_16k_kv4.json"
log="$OUTDIR/pilot_realprefill_32cachelru_16k_kv4.log"

echo "[run_pilot_inner] === preset=$PRESET 1-rep PILOT starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" \
  --contexts 16384 \
  --output-tokens 256 \
  --max-live-kv-tokens 16640 \
  --kv-quant q4 \
  --hy3-depths 3 \
  --output-json "$out" \
  >> "$log" 2>&1
rc=$?
echo "[run_pilot_inner] === preset=$PRESET 1-rep PILOT exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
echo "[run_pilot_inner] preset=$PRESET 1-rep PILOT exit=$rc $(date -u +%FT%TZ)" >&2

exit $rc
