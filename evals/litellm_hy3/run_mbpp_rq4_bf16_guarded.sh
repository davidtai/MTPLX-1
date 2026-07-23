#!/usr/bin/env bash
# OUTER launcher: MBPP full-974 arm of record — champion rq4 + bf16 KV.
#
# T2 quality arm of record (2026-07-23): oq2e weights + proj-requant q4
# (rq4) + bf16 KV. kv4 was REVERSED as serving mode 2026-07-22 (MTP dead
# under kv4 at real prefill). This is the canonical shipping config.
#
# Runtime identity (consumed by evals/litellm_hy3/handler.py::_champion_overrides):
#   MODEL_KEY   = hy3-expert-oq2e
#   MODEL_ROOT  = ~/.cache/huggingface/hy3-oq2e-mlx
#   ISLAND_LAYER_COUNT = 79 (full residency, fastest serving config)
#   PROJ_QUANT  = none (pre-quantized residents; forcing raises)
#   PROJ_REQUANT = q4  (re-quantizes trunk *_proj residents to q4/gs64)
#   KV_QUANT    = <unset/none> => bf16 trunk KV (champion baseline, NOT kv4)
#
# Suite: MBPP, 974 tasks (upstream-broken: MBPP/180 + MBPP/493), greedy
# (temp=0), seed 42, max-tokens 1024, chat endpoint, workers=4.
# Wall budget: ~60 min (q8 ref took 55.7 min at comparable load). Child
# timeout = 7200 s (2 h) for safety margin.
#
# Box laws: GPU only inside guarded window (scripts/run_with_qwen_stopped.py
# holds exclusive flock + stops qwen). Launch with Bash run_in_background:true,
# STANDALONE (never shell-&, never chained, no Bash timeout parameter).
# Export PYTHONPATH=EVALWT:CAMPSP before exec (run_with_qwen_stopped.py
# imports mtplx.qwen_guard from this Python env).
set -uo pipefail

ISOPY="/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/5bc9942c-42b5-4300-bc57-4b4f43ee476e/scratchpad/litellm-serve-venv/bin/python"
EVALWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
CAMPSP="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages"
# run_with_qwen_stopped.py itself imports mtplx.qwen_guard, so PYTHONPATH must
# be set for THIS process before exec (not just inherited by the inner child).
export PYTHONPATH="$EVALWT:$CAMPSP"

# Champion runtime identity (rq4 + bf16 KV).
export MTPLX_HY3_MODEL_KEY="hy3-expert-oq2e"
export MTPLX_HY3_MODEL_ROOT="$HOME/.cache/huggingface/hy3-oq2e-mlx"
export MTPLX_HY3_ISLAND_LAYER_COUNT="79"
export MTPLX_HY3_PROJ_QUANT="none"          # pre-quantized residents; forcing raises
export MTPLX_HY3_PROJ_REQUANT="q4"          # rq4: re-quantize trunk *_proj to q4/gs64
# MTPLX_HY3_KV_QUANT intentionally NOT SET => bf16 KV (champion baseline)
# kv4 is REVERSED as serving mode (MTP dead under kv4 at real prefill, 2026-07-22)

# Output paths for this arm.
export MBPP_OUTPUT="$EVALWT/evals/tier2/mbpp_oq2e_rq4_bf16_full974.json"
export MTPLX_HY3_PROXY_LOG="$EVALWT/evals/tier2/mbpp_rq4_bf16_proxy.log"
export MTPLX_LITELLM_PY="$ISOPY"

PLIST="${QWEN_PLIST:-$HOME/Library/LaunchAgents/com.tea.qwen.plist}"
LOCK_PATH="${MTPLX_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
QWEN_API_URL="${QWEN_API_URL:-http://127.0.0.1:8080/v1/models}"
LOCK_TIMEOUT="${MTPLX_LOCK_TIMEOUT:-7200}"

exec "$ISOPY" "$EVALWT/scripts/run_with_qwen_stopped.py" \
  --plist "$PLIST" \
  --api-url "$QWEN_API_URL" \
  --lock-path "$LOCK_PATH" \
  --lock-timeout-seconds "$LOCK_TIMEOUT" \
  --child-timeout-seconds 7200 \
  -- \
  bash "$EVALWT/evals/litellm_hy3/serve_and_eval_mbpp.sh"
