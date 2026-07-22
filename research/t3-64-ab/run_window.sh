#!/usr/bin/env bash
# OUTER launcher: T3 64 GiB envelope island-vs-cache A/B (2026-07-22).
#
# Three parts, ONE guarded window:
#   1. Arm A   -- preset hy3-oq2e-rq4-64 (islands 52), 16k KV, AR+K1+K2+K3, 3 reps.
#                 Doubles as the official T3 matrix 64x16k bf16 cell receipt.
#   2. Arm B   -- same 64 GiB / 16k-KV envelope, ZERO islands, freed budget
#                 reallocated to an explicitly-sized expert-cache (derived via
#                 research/t3-64-ab/preflight.py). Two cache-policy sub-arms
#                 (frequency, lru), AR+K1+K2+K3 each, 3 reps.
#   3. Patch   -- Arm A's config at 32k KV / 79.75 GiB (the SAME override
#                 research/t3-64x32k already GPU-verified), K3 only, 3 reps --
#                 cures the bit-exact caveat on the existing 64x32k cell
#                 receipt (that run had no MTPLX_HY3_ROUTER_SPLITK_M1=all).
#
# CPU-only preflight (research/t3-64-ab/preflight.py) MUST pass before the
# guarded window opens -- if it rejects (any lane REJECTs, or the manifest
# root-vs-manifest diff doesn't match the expected 2-extra/0-missing state),
# this script exits 2 and NO GPU work happens, no qwen stop, no flock
# acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a STANDALONE
# command (never chained, never shell `&`) -- box law: a chained window died
# to a hook timeout on 2026-07-21.
set -uo pipefail

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_window] CPU-only admission preflight (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-64-ab/preflight.py \
  > "$WT/evals/tier2/t3_64_ab_admission_preflight.json" \
  2> "$WT/evals/tier2/t3_64_ab_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_window] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/t3_64_ab_admission_preflight.log" >&2
  exit 2
fi
echo "[run_window] preflight OK; all 4 lane configs ADMIT, manifest diff matches expected state" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 18000 \
  -- \
  bash "$WT/research/t3-64-ab/run_window_inner.sh"
