#!/usr/bin/env bash
# OUTER launcher: T3 32-envelope-column real-prefill matrix guarded window
# (2026-07-22). Matrix K locked = K1 (2e2ed99, #130 comment 5048661802).
#
# Preset `hy3-oq2e-rq4-32-cachelru` (ZERO islands, cache-heavy LRU -- David's
# <=64 GiB no-islands directive). Four arms in ONE guarded window (one model
# load per rep, one qwen-stop/flock hold covering all twelve process
# invocations): 32x1024, 32x16k, 32x64k, naturalistic-32. All real long-code
# prefill (except the naturalistic arm's short natural-language tail),
# decode 256, greedy, bf16 KV (--kv-quant omitted), K1 (--hy3-depths 1; AR
# row auto-runs alongside for free), 3 reps each.
#
# Memory-limit overrides (harness-derived by research/t3-32col/preflight.py,
# never hand-derived):
#   1024 / 16384 / naturalistic arms: no override. The preset's own declared
#     39 GiB limit ADMITs at all three (its ~18 GiB elastic cache shrinks to
#     14.4509 GiB to absorb the 16384 arm's KV growth, margin 0.0799 GiB;
#     1024/naturalistic land at the preset's own 4096-token default, margin
#     0.3929 GiB).
#   65536 arm: --max-live-kv-tokens 65792, --memory-limit 62092476416
#     (57.8281 GiB; fixed=39.4693 GiB, cache=17.9659 GiB, margin 0.3929 GiB).
#     UNLIKE the 64/48-envelope siblings, this is the column where the
#     shrinking-cache trick finally runs out: the preset's own declared
#     39 GiB limit REJECTs here by 503,861,760 bytes (the naturally-sized
#     cache would need to go negative) -- the family KV-delta override
#     formula (declared_limit + (kv_tokens-4096)*327,680) recovers ADMIT at
#     57.8281 GiB, well under the 112 GiB wired knob and the 90 GiB sign-off
#     ceiling.
#
# Sign-off (David's rule 7): every derived limit here (39 / 57.8281 GiB)
# sits well under the sub-85/90 GiB sign-off band and the 112 GiB wired
# knob -- no fresh ask needed for this column.
#
# CPU-only preflight (research/t3-32col/preflight.py) MUST pass (all
# runnable arms ADMIT; any arm whose derived override would exceed the
# 112 GiB knob, or that still REJECTs at that override, is marked SKIP and
# excluded) before this guarded window opens -- if it fails, this script
# exits 2 and NO GPU work happens, no qwen stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_32col] wired-knob (recorded, not gated at this margin): sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "[run_32col] iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" | tee "$WT/evals/tier2/t3_32col_wired_knob.txt" >&2

echo "[run_32col] CPU-only admission preflight (zero GPU touch) for all four arms..." >&2
PYTHONPATH="$WT" "$PY" research/t3-32col/preflight.py \
  --output "$WT/evals/tier2/t3_32col_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_32col_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_32col] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_32col_admission_preflight.log" >&2
  exit 2
fi
echo "[run_32col] preflight OK; all runnable arms ADMIT" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 10800 \
  --child-timeout-seconds 10800 \
  -- \
  bash "$WT/research/t3-32col/run_32col_inner.sh"
