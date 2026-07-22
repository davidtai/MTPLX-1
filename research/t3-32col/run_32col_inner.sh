#!/usr/bin/env bash
# INNER command for the T3 32-envelope-column real-prefill matrix guarded
# window (2026-07-22).
#
# The CALLER (run_32col.sh -> run_with_qwen_stopped.py) already holds the
# exclusive MLX flock and has stopped qwen. Runs FOUR arms SERIALLY, 3 reps
# each (one model resident at a time, per box law; memory fully releases
# between the 12 process invocations):
#
#   ARM 1: hy3-oq2e-rq4-32-cachelru x contexts=1024  x output-tokens=256,
#          kv left at the preset's own 4096-token default (no override).
#   ARM 2: same preset x contexts=16384 x output-tokens=256,
#          --max-live-kv-tokens 16640. NO --memory-limit override needed --
#          research/t3-32col/preflight.py confirms this preset's own
#          declared 39 GiB limit ADMITs here (its ~18 GiB elastic cache
#          shrinks to 14.4509 GiB to absorb the KV growth, margin 0.0799
#          GiB).
#   ARM 3: same preset x contexts=65536 x output-tokens=256,
#          --max-live-kv-tokens 65792, --memory-limit 62092476416
#          (57.8281 GiB, harness-derived by research/t3-32col/preflight.py).
#          UNLIKE the 64/48-envelope siblings, this is the column where the
#          shrinking-cache trick runs out: the preset's own declared 39 GiB
#          limit REJECTs here by 503,861,760 bytes (cache would need to go
#          negative) -- the family KV-delta override formula
#          (declared_limit + (kv_tokens-4096)*327,680) recovers ADMIT at
#          57.8281 GiB, well under the 112 GiB wired knob and the 90 GiB
#          sign-off ceiling.
#   ARM 4: naturalistic ~265-token non-code tail (research/t3-32col/
#          run_naturalistic_32_rep.py), same preset, kv left at default.
#
# All arms: --hy3-depths 1 (K1; AR/depth=0 auto-runs alongside for free),
# bf16 KV (--kv-quant omitted throughout), greedy (harness default).
# Preset hy3-oq2e-rq4-32-cachelru = ZERO islands, all 79 routed layers
# stream through one shared LRU cache (David's <=64 GiB no-islands
# directive) -- expect REAL streaming telemetry (decode_expert_cache_hit_rate
# populated and non-null on every row).
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all -- process-wide, all reps,
# all arms, same convention as every other t3-* window. MTPLX_SUSTAINED_PREFILL
# and MTPLX_HY3_SUBMIT_CADENCE are ALSO this preset's own declared env table
# ([preset.hy3-oq2e-rq4-32-cachelru.env] in benchmarks/presets.toml,
# apply_preset_defaults() only os.environ.setdefault()s them anyway) -- kept
# explicit here for parity with every other t3-* window.
#
# Usage (never invoked directly -- always via run_32col.sh):
#   run_32col_inner.sh
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
  echo "[run_32col_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps, all arms.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# Mirrors [preset.hy3-oq2e-rq4-32-cachelru.env] in benchmarks/presets.toml.
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

MEMLIMIT_64K=62092476416  # 57.8281 GiB; harness-derived, see preflight.py

overall_rc=0

echo "[run_32col_inner] === ARM 1 (32x1024) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_32col_1024_rep${rep}.json"
  log="$OUTDIR/t3_32col_1024_rep${rep}.log"
  echo "[run_32col_inner] === arm=1024 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 1024 \
    --output-tokens 256 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_32col_inner] === arm=1024 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_32col_inner] arm=1024 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_32col_inner] === ARM 2 (32x16k) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_32col_16384_rep${rep}.json"
  log="$OUTDIR/t3_32col_16384_rep${rep}.log"
  echo "[run_32col_inner] === arm=16384 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 16384 \
    --output-tokens 256 \
    --max-live-kv-tokens 16640 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_32col_inner] === arm=16384 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_32col_inner] arm=16384 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_32col_inner] === ARM 3 (32x64k) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_32col_65536_rep${rep}.json"
  log="$OUTDIR/t3_32col_65536_rep${rep}.log"
  echo "[run_32col_inner] === arm=65536 preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "$PRESET" \
    --contexts 65536 \
    --output-tokens 256 \
    --max-live-kv-tokens 65792 \
    --memory-limit "$MEMLIMIT_64K" \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_32col_inner] === arm=65536 preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_32col_inner] arm=65536 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_32col_inner] === ARM 4 (naturalistic ~265) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_32col_naturalistic_rep${rep}.json"
  log="$OUTDIR/t3_32col_naturalistic_rep${rep}.log"
  echo "[run_32col_inner] === arm=naturalistic preset=$PRESET rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" research/t3-32col/run_naturalistic_32_rep.py \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_32col_inner] === arm=naturalistic preset=$PRESET rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_32col_inner] arm=naturalistic rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_32col_inner] all reps attempted (all four arms); overall_rc=$overall_rc" >&2
exit $overall_rc
