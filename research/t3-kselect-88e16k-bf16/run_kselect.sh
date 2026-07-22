#!/usr/bin/env bash
# OUTER launcher: T3 **bf16-KV** K-SELECTION reference guarded window
# (2026-07-22). This is the follow-on grant after the kv4 K-selection cell
# of record (`db46f26`, #130 comment 5047822076, decode AR 26.2 / K1 16.5 /
# K2 13.6 / K3 10.8 -- MTP dead under kv4 at real prefill). Same shape,
# bf16 KV instead of kv4 (kv4 was reversed as serving mode 2026-07-22): this
# window isolates kv4's acceptance cost at real prefill by re-running the
# identical 88e x 16k AR+K1/K2/K3 x3-rep shape with KV precision at bf16
# (--kv-quant omitted -- the CLI flag's own default is `None`, i.e. off/
# bf16; scripts/benchmark_q2_mtp_depth_matrix.py's `--kv-quant` argparse
# definition has `default=None`, confirmed by reading the flag, not
# assumed).
#
# Preset `hy3-oq2e-rq4-88e` (efficient-88, committed c513c6d), REAL prefill
# (--contexts 16384), decode 256, --max-live-kv-tokens 16640 (16384 +
# 256 output), greedy (harness default), AR + K1/K2/K3
# (--hy3-depths 1,2,3 -- AR/depth=0 auto-runs regardless), 3 reps (3
# independent process invocations -- retained_replicates is hardcoded 1
# inside the harness, no --reps flag exists; same convention as the kv4
# sibling window research/t3-kselect-88e16k/run_kselect_inner.sh).
#
# MEMORY-LIMIT OVERRIDE REQUIRED (unlike the kv4 cell): bf16 KV is
# 327,680 B/token vs kv4's 92,160 B/token (3.556x). At kv=16640 the preset's
# own declared 96 GiB limit REJECTs by 3.4575 GiB (fixed=99.4575 GiB). The
# EXACT override is derived by research/t3-kselect-88e16k-bf16/preflight.py
# via the harness (never hand-derived): declared_limit_bytes(103,079,215,104
# = 96 GiB) + (16640-4096)*327,680 = 107,189,633,024 B = 99.8281 GiB --
# confirmed ADMIT with the family's own constant +0.3706 GiB margin (same
# margin the -88e preset carries at its own 4096-token control point; see
# benchmarks/presets.toml's -88e description). This is well under the
# 112 GiB (114688 MiB) wired knob David set for this window.
#
# WIRED-KNOB PREFLIGHT (mandatory, David 2026-07-22): sysctl
# iogpu.wired_limit_mb MUST read 114688 before this script proceeds past
# its own preflight -- checked below; if it does not, this script exits 3
# and NO GPU work happens, no qwen stop, no flock acquired.
#
# Expected hard peak ~96-97 GiB (kv4 sibling peaked 93.0 GiB; bf16 16640-
# token KV adds ~3.6 GiB of resident KV bytes over kv4's). At the 112 GiB
# knob the known failure mode is SILENT macOS memory starvation (no panic
# log) -- nothing else big may run census-side; peak is watched in-log
# while polling.
#
# Sign-off (David's rule 7): projected peak ~96-97 GiB is >85 GiB; this
# bf16-KV reference pass was David's own live grant 2026-07-22 immediately
# following the kv4 K-selection cell -- covered, not asking again.
#
# exact-lane env MTPLX_HY3_ROUTER_SPLITK_M1=all is exported inside
# run_kselect_inner.sh (in NO preset, must be set manually, read once per
# process -- see run_kselect_inner.sh header, same convention as t3-80/
# t3-88/t3-kselect-88e16k(kv4)/t3-pilot-realprefill).
#
# CPU-only preflight (research/t3-kselect-88e16k-bf16/preflight.py) MUST
# pass (derived-override ADMIT) before the guarded window opens -- if it
# REJECTs, this script exits 2 and NO GPU work happens, no qwen stop, no
# flock acquired.
#
# Launch this via the Bash tool with run_in_background:true, as a
# STANDALONE command (never chained, never shell `&`, never with a Bash
# `timeout` parameter) -- box law: an out-of-range timeout killed a window
# once.
set -uo pipefail

PRESET="hy3-oq2e-rq4-88e"
KV=16640
EXPECTED_WIRED_LIMIT_MB=114688

WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
cd "$WT" || exit 9

echo "[run_kselect_bf16] wired-knob preflight: checking sysctl iogpu.wired_limit_mb..." >&2
ACTUAL_WIRED_LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
if [[ "$ACTUAL_WIRED_LIMIT_MB" != "$EXPECTED_WIRED_LIMIT_MB" ]]; then
  echo "[run_kselect_bf16] WIRED-KNOB MISMATCH: expected $EXPECTED_WIRED_LIMIT_MB, got '$ACTUAL_WIRED_LIMIT_MB' -- NOT launching, no GPU touch, no qwen stop" >&2
  exit 3
fi
echo "[run_kselect_bf16] wired-knob OK: iogpu.wired_limit_mb=$ACTUAL_WIRED_LIMIT_MB" >&2

echo "[run_kselect_bf16] CPU-only admission preflight (two-phase, derives exact bf16 KV-delta override) for $PRESET @ kv=$KV bf16 (zero GPU touch)..." >&2
PYTHONPATH="$WT" "$PY" research/t3-kselect-88e16k-bf16/preflight.py --kv "$KV" \
  > "$WT/evals/tier2/kselect_88e16k_bf16_admission_preflight.json" \
  2> "$WT/evals/tier2/kselect_88e16k_bf16_admission_preflight.log"
PREFLIGHT_RC=$?
if [[ "$PREFLIGHT_RC" != 0 ]]; then
  echo "[run_kselect_bf16] PREFLIGHT FAILED (rc=$PREFLIGHT_RC) -- not opening a window" >&2
  cat "$WT/evals/tier2/kselect_88e16k_bf16_admission_preflight.log" >&2
  exit 2
fi
echo "[run_kselect_bf16] preflight OK; $PRESET ADMITs at kv=$KV bf16 with the derived override" >&2

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 7200 \
  --child-timeout-seconds 28800 \
  -- \
  bash "$WT/research/t3-kselect-88e16k-bf16/run_kselect_inner.sh"
