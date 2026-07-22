#!/usr/bin/env bash
# OUTER launcher, WINDOW 2 (run only AFTER window 1 / run_speed_arms.sh has
# fully completed and released the flock -- never overlapping windows):
# HumanEval n=20 screen for the kv4 arm of the T3-pre 64x32k cell.
#
# Same runtime identity as the kv4 speed arm (preset hy3-oq2e-rq4-64:
# islands 52, proj_requant q4, hy3-router-kernel
# mpp-fp32-splitk-r1-fused-r2, cache-policy frequency/layer/component-banks,
# expert-cache-limit 2GiB, runtime-reserve 7GiB, expert-integrity
# headers-only, split-route-release deferred) plus --kv-quant q4, reached
# through evals/litellm_hy3/handler.py's MTPLX_HY3_* env overrides (the
# handler's own DEFAULT_RUNTIME_OPTIONS + _champion_overrides() layering is
# reused unmodified; this lane's only code change is adding kv_quant support
# to _champion_overrides, mirroring the existing proj_requant "none" sentinel
# pattern -- see git log on evals/litellm_hy3/handler.py). Same 79.75 GiB
# memory-limit override as the speed arms (comparability: a shared ceiling,
# not sized per-arm) and the SAME 32768-token KV admission envelope, so this
# is genuinely the kv4 arm's identity, just serving HumanEval prompts
# instead of the synthetic 1024/1024 shape.
#
# Run as its OWN guarded window (not fused with the 3-arm speed sweep): every
# prior HumanEval gate in this campaign (evals/tier2/NOTES.md: shipped-q2,
# oQ2e, proj_requant 20-task, proj_requant full-164) was launched as its own
# dedicated window, never combined with a depth-matrix speed sweep -- adding
# a 4th heavy load (a different serving stack: litellm proxy + handler.py)
# onto an already multi-hour 3-arm window raises the failure surface for no
# offsetting benefit, so this follows the established pattern instead.
#
# Launch via the Bash tool with run_in_background:true, standalone (never
# chained, never shell `&`), same as window 1.
set -uo pipefail

ISOPY="/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/5bc9942c-42b5-4300-bc57-4b4f43ee476e/scratchpad/litellm-serve-venv/bin/python"
EVALWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
CAMPSP="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages"
# run_with_qwen_stopped.py itself (not just the inner serve_and_eval.sh)
# imports mtplx.qwen_guard, so PYTHONPATH must be set for THIS process
# before exec, not merely inherited by the child. (First launch attempt
# failed here: ModuleNotFoundError: No module named 'mtplx' from the
# wrapper's own import line, before qwen was ever touched -- no collateral.)
export PYTHONPATH="$EVALWT:$CAMPSP"

# The kv4 arm's exact runtime identity -- consumed by
# evals/litellm_hy3/handler.py::_champion_overrides().
export MTPLX_HY3_MODEL_KEY="hy3-expert-oq2e"
export MTPLX_HY3_MODEL_ROOT="$HOME/.cache/huggingface/hy3-oq2e-mlx"
export MTPLX_HY3_ISLAND_LAYER_COUNT="52"
export MTPLX_HY3_PROJ_QUANT="none"          # residents pre-quantized; forcing raises
export MTPLX_HY3_PROJ_REQUANT="q4"
export MTPLX_HY3_KV_QUANT="q4"
export MTPLX_HY3_ROUTER_KERNEL="mpp-fp32-splitk-r1-fused-r2"
export MTPLX_HY3_MEMORY_LIMIT="85630910464"     # 79.75 GiB, same override as the speed arms
export MTPLX_HY3_MAX_LIVE_KV_TOKENS="32768"     # same admission envelope as the kv4 speed arm

export HUMANEVAL_OUTPUT="$EVALWT/evals/tier2/humaneval_t3_64x32k_kv4_n20.json"
export MTPLX_HY3_PROXY_LOG="$EVALWT/evals/tier2/humaneval_t3_64x32k_kv4_n20_proxy.log"
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
  --child-timeout-seconds 3600 \
  -- \
  bash "$EVALWT/evals/litellm_hy3/serve_and_eval.sh"
