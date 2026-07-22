#!/usr/bin/env bash
# OUTER launcher: T3 bf16-KV K-SELECTION window, REP 2 ONLY (recovery run,
# 2026-07-22).
#
# The original 3-rep window (run_kselect.sh) had its background wrapper
# process (and rep2's benchmark_q2_mtp_depth_matrix.py child) terminated
# ~7 minutes into rep 2 (10:29:33-10:36:39 CDT). Root-cause investigated
# (log show, DiagnosticReports, vm_stat/vm.swapusage) -- NOT a box/memory/
# GPU fault: no crash report generated anywhere near the exit timestamp, no
# jetsam/memorystatus kill naming the PID, clean swap/no memory pressure
# throughout. The exit log shows `python3.12: (CoreAnalytics) Entering exit
# handler` at the exact termination instant -- this atexit-style signature
# only fires on a graceful-ish signal-driven unwind (SIGTERM/SIGINT), never
# on SIGKILL/a hard fault. Best-evidence conclusion: a driver-session
# interrupt (unrelated foreground tool-call rejection) propagated a stop
# signal to the whole session process group, taking the background GPU
# window down as collateral, not a genuine run failure. Rep 1 completed
# cleanly and PASSED (evals/tier2/t3_kselect_88e16k_bf16_rep1.json,
# hard_peak 92.95 GiB, kv_quant=None/bf16 confirmed, all 4 lanes real
# 16384-token prefill). This script retries ONLY the missing rep (2), not
# re-running rep 1, matching the exact recovery precedent used by the kv4
# sibling window (research/t3-kselect-88e16k/run_kselect_rep3.sh) when it
# hit an analogous mid-rep cutoff.
#
# Same exact shape as the original window: hy3-oq2e-rq4-88e, --contexts
# 16384 --output-tokens 256 --max-live-kv-tokens 16640, bf16 KV
# (--kv-quant omitted), --hy3-depths 1,2,3, greedy (harness default), with
# the SAME harness-derived --memory-limit override
# (107,189,633,024 B = 99.8281 GiB; see preflight.py) -- re-verified below,
# not re-derived from scratch, since the shape is unchanged.
#
# WIRED-KNOB PREFLIGHT (mandatory): sysctl iogpu.wired_limit_mb MUST read
# 114688 before this proceeds -- checked below; exits 3 with no GPU touch
# if it does not.
#
# CPU-only admission preflight (research/t3-kselect-88e16k-bf16/
# preflight.py) MUST pass (derived-override ADMIT) before the guarded
# window opens -- if it REJECTs, this script exits 2, no GPU work, no qwen
# stop, no flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter).
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
KV=16640
EXPECTED_WIRED_LIMIT_MB=114688

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_kselect_rep2_bf16] wired-knob preflight..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
if [[ "$ACTUAL_WIRED_LIMIT_MB" != "$EXPECTED_WIRED_LIMIT_MB" ]]; then
  echo "[run_kselect_rep2_bf16] WIRED-KNOB MISMATCH: expected $EXPECTED_WIRED_LIMIT_MB, got '$ACTUAL_WIRED_LIMIT_MB' -- NOT launching" >&2
  exit 3
fi
echo "[run_kselect_rep2_bf16] wired-knob OK: iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" >&2

echo "[run_kselect_rep2_bf16] CPU-only admission preflight for $PRESET @ kv=$KV bf16 (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-kselect-88e16k-bf16/preflight.py --kv "$KV" \
  > "$WT/evals/tier2/kselect_88e16k_bf16_rep2_admission_preflight.json" \
  2> "$WT/evals/tier2/kselect_88e16k_bf16_rep2_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_kselect_rep2_bf16] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/kselect_88e16k_bf16_rep2_admission_preflight.log" >&2
  exit 2
fi
echo "[run_kselect_rep2_bf16] preflight OK; $PRESET ADMITs at kv=$KV bf16 with the derived override" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 3600 \
  -- \
  bash "$WT/research/t3-kselect-88e16k-bf16/run_kselect_rep2_inner.sh"
