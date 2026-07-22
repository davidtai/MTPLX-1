#!/usr/bin/env bash
# INNER command for the T3 **bf16-KV** K-SELECTION reference guarded window
# (2026-07-22).
#
# The CALLER (run_kselect.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen. Runs 3 reps SERIALLY (one model
# resident at a time, per box law); each rep is its own
# benchmark_q2_mtp_depth_matrix.py process (memory fully releases between
# reps). --hy3-depths 1,2,3 requests K1/K2/K3; AR (depth=0) auto-runs
# regardless of what --hy3-depths requests, so every rep yields all four
# lanes (AR, K1, K2, K3) in one process/one real 16384-token prefill.
#
# Real prefill: --contexts 16384 builds an exactly-16384-token long-code
# prompt with hard length gates (benchmark_q2_mtp_depth_matrix.py:2961-3005).
# bf16 KV: --kv-quant is OMITTED entirely (the flag's own default is None,
# i.e. off/bf16 -- confirmed by reading the argparse definition, not
# assumed). Greedy decoding is the harness default (no --temperature/
# --sampling flag passed).
#
# MEMORY-LIMIT OVERRIDE (required at bf16, unlike the kv4 sibling window):
# preset hy3-oq2e-rq4-88e declares memory-limit=96GiB, which REJECTs bf16
# KV at 16640 live-KV tokens by 3.4575 GiB (bf16 is 327,680 B/token vs
# kv4's 92,160 B/token). Exact override derived by
# research/t3-kselect-88e16k-bf16/preflight.py (CPU-only, harness-verified,
# two-phase: read declared_limit_bytes + fixed_bytes with no override, then
# re-gate WITH the derived override and require ADMIT):
#   override_bytes = declared_limit_bytes(103,079,215,104 = 96 GiB)
#                     + (16640 - 4096) * 327,680 B/token
#                     = 107,189,633,024 B = 99.8281 GiB
#   (fixed_bytes 99.4575 GiB, margin +0.3706 GiB -- the same constant margin
#   the -88e preset family carries at every KV level; see preflight JSON
#   evals/tier2/kselect_88e16k_bf16_admission_preflight.json for the full
#   two-phase receipt.) Well under the 112 GiB (114688 MiB) wired knob.
#
# exact-lane env: MTPLX_HY3_ROUTER_SPLITK_M1=all is exported globally for
# every rep in this window -- same convention as research/t3-80/
# research/t3-88/research/t3-kselect-88e16k(kv4)/t3-pilot-realprefill (the
# router kernel is configured once at model load inside mtplx/runtime.py's
# _load_impl, os.environ.get("MTPLX_HY3_ROUTER_SPLITK_M1", "mtp"), read a
# single time per process; applies process-wide to every K arm in that rep,
# including the AR lane's own parity).
#
# Usage (never invoked directly -- always via run_kselect.sh):
#   run_kselect_inner.sh
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
MEMLIMIT=107189633024  # 99.8281 GiB; harness-derived, see preflight.py

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
  echo "[run_kselect_inner_bf16] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# apply_preset_defaults() only os.environ.setdefault()s these from the
# preset's own [preset.hy3-oq2e-rq4-88e.env] table anyway, so exporting here
# is belt-and-braces, not strictly required -- kept for parity with the
# t3-80 / t3-88 / t3-kselect-88e16k(kv4) convention.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

overall_rc=0
for rep in 1 2 3; do
  out="$OUTDIR/t3_kselect_88e16k_bf16_rep${rep}.json"
  log="$OUTDIR/t3_kselect_88e16k_bf16_rep${rep}.log"
  echo "[run_kselect_inner_bf16] === preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 16384 \
    --output-tokens 256 \
    --max-live-kv-tokens 16640 \
    --memory-limit "$MEMLIMIT" \
    --hy3-depths 1,2,3 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_kselect_inner_bf16] === preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_kselect_inner_bf16] preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_kselect_inner_bf16] all reps attempted; overall_rc=$overall_rc" >&2
exit $overall_rc
