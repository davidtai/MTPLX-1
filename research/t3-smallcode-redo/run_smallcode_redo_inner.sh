#!/usr/bin/env bash
# INNER command for the T3 small-code arm-of-record guarded window
# (2026-07-22).
#
# contexts=320: minimum viable N that preserves the standard coding-agent
# tail (mtplx/prefill_bench.py DEFAULT_FINAL_REQUEST, no custom prompt_tail).
# contexts=256 FAILED the release gate 2026-07-22 -- the fixed "Final user
# request" tail chat-templates to 319 tokens on hy3-oq2e (prompt_style=
# "coding-agent", prompt_format="chat", enable_thinking=False, the exact
# options _run_depth_matrix_impl always uses), so at 256 the
# `len(tail_ids) >= context_tokens` truncation branch fires
# (prefill_bench.py:199-217), stamping prompt_release_valid=False and
# prompt_tail_preserved=False, and the release gate
# (scripts/benchmark_q2_mtp_depth_matrix.py:2992-3004) raises
# BenchmarkGateError -- all 9 reps failed at contexts=256 in ~15s each
# (see evals/tier2/t3_smallcode_*_rep*.failed-ctx256.json/.log). Direct-probed
# CPU-only (same method documented in research/t3-88col-1024-nat/
# run_naturalistic_rep.py's header for the custom naturalistic tail,
# replicated here for the STANDARD builder): ascending N from 257, first
# N where prompt_release_valid=True AND prompt_tail_preserved=True is
# N=320 (== tail length 319 + 1 structural filler token, same shape as the
# naturalistic 264+1=265 case). David's "256 is fine" is honored as close
# as the gate allows -- 320 is the smallest value that clears it.
#
# The CALLER (run_smallcode_redo.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen. Runs THREE arms
# SERIALLY, 3 reps each (one model resident at a time, per box law; memory
# fully releases between the 9 process invocations):
#
#   ARM 88e: hy3-oq2e-rq4-88e x contexts=320 x output-tokens=256,
#            kv left at the preset's own 4096-token default (no override).
#            Full residency (islands 79, 0 streamed) -- decode_expert_cache_hit_rate
#            is EXPECTED to be null/absent here, same as every other 88e
#            window (t3_88col_1024_rep*, t3_88col_naturalistic_rep*).
#   ARM 80:  hy3-oq2e-rq4-80 x contexts=320 x output-tokens=256, same kv
#            default (no override -- research/t3-smallcode-redo/preflight.py
#            confirms this preset's own declared 87 GiB limit ADMITs here,
#            margin +0.0223 GiB, same control point as every other 80-column
#            "1024" arm). Islands 69 (10 streamed) -- expect real
#            decode_expert_cache_hit_rate telemetry.
#   ARM 64:  hy3-oq2e-rq4-64-cachelru x contexts=320 x output-tokens=256,
#            same kv default (no override -- declared 71 GiB limit ADMITs,
#            margin +0.3667 GiB). ZERO islands (all 79 routed layers stream
#            through one shared LRU cache) -- expect real
#            decode_expert_cache_hit_rate telemetry on every row.
#
# All arms: plain `--contexts 320` against the SAME release-valid
# realistic_programming_v1 builder every other matrix cell already uses --
# NO custom prompt_tail, NO library-call mechanics/monkeypatching (unlike
# research/t3-*/run_naturalistic_*_rep.py, whose only reason to exist is
# hy3-oq2e's lack of a --prompt-tail CLI flag for genuinely custom prose;
# not needed here). --hy3-depths 1 (K1; AR/depth=0 auto-runs alongside for
# free), bf16 KV (--kv-quant omitted throughout), greedy (harness default).
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all -- process-wide, all reps,
# all arms, same convention as every other t3-* window. MTPLX_SUSTAINED_PREFILL
# and MTPLX_HY3_SUBMIT_CADENCE are ALSO each preset's own declared env table
# ([preset.<name>.env] in benchmarks/presets.toml, apply_preset_defaults()
# only os.environ.setdefault()s them anyway -- identical values across all
# three presets here) -- kept explicit for parity with every other t3-*
# window.
#
# Usage (never invoked directly -- always via run_smallcode_redo.sh):
#   run_smallcode_redo_inner.sh
set -uo pipefail

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
  echo "[run_smallcode_redo_inner] mtplx resolution assertion FAILED -- aborting before any GPU work" >&2
  exit 9
fi

# exact-lane convention (see header): process-wide, all reps, all arms.
export MTPLX_HY3_ROUTER_SPLITK_M1=all
# Mirrors each preset's own [preset.<name>.env] table (identical values
# across hy3-oq2e-rq4-88e / -80 / -64-cachelru).
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

overall_rc=0

echo "[run_smallcode_redo_inner] === ARM 88e (smallcode x320) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_smallcode_88e_rep${rep}.json"
  log="$OUTDIR/t3_smallcode_88e_rep${rep}.log"
  echo "[run_smallcode_redo_inner] === arm=88e preset=hy3-oq2e-rq4-88e rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "hy3-oq2e-rq4-88e" \
    --contexts 320 \
    --output-tokens 256 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_smallcode_redo_inner] === arm=88e preset=hy3-oq2e-rq4-88e rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_smallcode_redo_inner] arm=88e rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_smallcode_redo_inner] === ARM 80 (smallcode x320) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_smallcode_80_rep${rep}.json"
  log="$OUTDIR/t3_smallcode_80_rep${rep}.log"
  echo "[run_smallcode_redo_inner] === arm=80 preset=hy3-oq2e-rq4-80 rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "hy3-oq2e-rq4-80" \
    --contexts 320 \
    --output-tokens 256 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_smallcode_redo_inner] === arm=80 preset=hy3-oq2e-rq4-80 rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_smallcode_redo_inner] arm=80 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_smallcode_redo_inner] === ARM 64-cachelru (smallcode x320) starting $(date -u +%FT%TZ) ===" >&2
for rep in 1 2 3; do
  out="$OUTDIR/t3_smallcode_64_rep${rep}.json"
  log="$OUTDIR/t3_smallcode_64_rep${rep}.log"
  echo "[run_smallcode_redo_inner] === arm=64 preset=hy3-oq2e-rq4-64-cachelru rep=$rep starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
  "$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
    --preset "hy3-oq2e-rq4-64-cachelru" \
    --contexts 320 \
    --output-tokens 256 \
    --hy3-depths 1 \
    --output-json "$out" \
    >> "$log" 2>&1
  rc=$?
  echo "[run_smallcode_redo_inner] === arm=64 preset=hy3-oq2e-rq4-64-cachelru rep=$rep exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
  echo "[run_smallcode_redo_inner] arm=64 rep=$rep exit=$rc $(date -u +%FT%TZ)" >&2
  [[ $rc -eq 0 ]] || overall_rc=$rc
done

echo "[run_smallcode_redo_inner] all reps attempted (all three arms); overall_rc=$overall_rc" >&2
exit $overall_rc
