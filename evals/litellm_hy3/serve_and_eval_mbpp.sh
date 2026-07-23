#!/usr/bin/env bash
# INNER command for the guarded MBPP-974 window (rq4 + bf16 KV arm of record).
#
# The CALLER (run_mbpp_rq4_bf16_guarded.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen. This script therefore
# runs entirely INSIDE the lock: it starts the litellm proxy, waits for it to
# be up, then runs code_eval_gate --suite mbpp against all 974 tasks, then
# tears the proxy down. No GPU work escapes the lock.
#
# Champion arm identity (T2 MBPP arm of record):
#   MTPLX_HY3_MODEL_KEY=hy3-expert-oq2e
#   MTPLX_HY3_MODEL_ROOT=~/.cache/huggingface/hy3-oq2e-mlx
#   MTPLX_HY3_ISLAND_LAYER_COUNT=79 (full residency)
#   MTPLX_HY3_PROJ_QUANT=none      (pre-quantized residents; raising raises)
#   MTPLX_HY3_PROJ_REQUANT=q4      (rq4: re-quantizes trunk *_proj to q4/gs64)
#   MTPLX_HY3_KV_QUANT unset/none  (bf16 KV -- NOT kv4; kv4 REVERSED 2026-07-22)
#
# Set via the outer launcher; this script just reads them.
set -uo pipefail

ISOPY="${MTPLX_LITELLM_PY:-/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/5bc9942c-42b5-4300-bc57-4b4f43ee476e/scratchpad/litellm-serve-venv/bin/python}"
EVALWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
CAMPSP="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages"
HERE="$EVALWT/evals/litellm_hy3"
export PYTHONPATH="$EVALWT:$CAMPSP"

PORT="${MTPLX_HY3_PORT:-18183}"
DATASET="${MBPP_DATASET:-/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/mbpp.jsonl}"
LIMIT="${MBPP_LIMIT:-974}"
OUT="${MBPP_OUTPUT:-$EVALWT/evals/tier2/mbpp_oq2e_rq4_bf16_full974.json}"
PROXY_LOG="${MTPLX_HY3_PROXY_LOG:-$EVALWT/evals/tier2/mbpp_rq4_bf16_proxy.log}"

# Champion decode env; memory knob stays inside ExpertStreamingConfig (103GiB),
# never a second env override.
unset MTPLX_MEMORY_LIMIT_BYTES || true
export MTPLX_SUSTAINED_PREFILL="${MTPLX_SUSTAINED_PREFILL:-1}"
export MTPLX_DEFERRED_PIN_RELEASE="${MTPLX_DEFERRED_PIN_RELEASE:-1}"
export MTPLX_HY3_SUBMIT_CADENCE="${MTPLX_HY3_SUBMIT_CADENCE:-8}"

mkdir -p "$(dirname "$OUT")"
[[ -f "$DATASET" ]] || { echo "[serve_and_eval_mbpp] dataset missing: $DATASET" >&2; exit 4; }

echo "[serve_and_eval_mbpp] starting litellm proxy on 127.0.0.1:$PORT (num_workers=1)" >&2
LITELLM_BIN="$(dirname "$ISOPY")/litellm"
"$LITELLM_BIN" --config "$HERE/config.yaml" --host 127.0.0.1 --port "$PORT" --num_workers 1 \
  > "$PROXY_LOG" 2>&1 &
PROXY=$!
trap 'kill "$PROXY" 2>/dev/null; wait "$PROXY" 2>/dev/null' EXIT

# Wait for the proxy to list the model (no model load yet).
up=0
for _ in $(seq 1 90); do
  if curl -s -m3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 'hy3-q2'; then up=1; break; fi
  kill -0 "$PROXY" 2>/dev/null || { echo "[serve_and_eval_mbpp] proxy died during startup; log:" >&2; tail -30 "$PROXY_LOG" >&2; exit 3; }
  sleep 2
done
[[ "$up" == 1 ]] || { echo "[serve_and_eval_mbpp] proxy never listed hy3-q2; log:" >&2; tail -30 "$PROXY_LOG" >&2; exit 5; }
echo "[serve_and_eval_mbpp] proxy up; running MBPP suite limit=$LIMIT (first request loads the model)" >&2

"$ISOPY" "$EVALWT/scripts/code_eval_gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model hy3-q2 \
  --suite mbpp \
  --dataset-path "$DATASET" \
  --limit "$LIMIT" \
  --endpoint chat \
  --max-tokens "${MBPP_MAX_TOKENS:-1024}" \
  --temperature "${MBPP_TEMPERATURE:-0.0}" \
  --seed "${MBPP_SEED:-42}" \
  --output-json "$OUT" \
  --progress \
  --allow-code-execution
RC=$?
echo "[serve_and_eval_mbpp] code_eval_gate exit=$RC" >&2
exit $RC
