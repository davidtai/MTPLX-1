#!/usr/bin/env bash
# INNER: HF-publish clean-room rehearsal on the SELECTIVE clone (residents +
# experts.bin + manifest + bundled layer80 head; expert safetensors ABSENT).
# Proves the published selective-download layout serves: resident load from
# model-resident-*, expert streaming from the bank only, MTP head picked up
# from the root itself (not hy3-bf16-and-mtp-layer80). K1 @ 1024 code
# context; acceptance ~0.89 expected per the matrix of record.
set -uo pipefail

PRESET="hy3-oq2e-rq4-64-cachelru"
SEL="/Users/davidtai/.cache/huggingface/hy3-oq2e-rehearsal-selective"
WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUTDIR="$WT/evals/tier2"
cd "$WT" || exit 9
export PYTHONPATH="$WT"

"$PY" -c "
import mtplx
assert mtplx.__file__.startswith('$WT'), mtplx.__file__
print('[assert] mtplx resolves to worktree:', mtplx.__file__)
" || exit 9

export MTPLX_HY3_ROUTER_SPLITK_M1=all
export MTPLX_SUSTAINED_PREFILL=1
export MTPLX_HY3_SUBMIT_CADENCE=8

out="$OUTDIR/publish_rehearsal_selective_64.json"
log="$OUTDIR/publish_rehearsal_selective_64.log"
echo "[publish-rehearsal] === selective-clone 64-envelope K1 starting $(date -u +%FT%TZ) ===" | tee "$log" >&2
"$PY" scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset "$PRESET" \
  --contexts 1024 \
  --output-tokens 128 \
  --hy3-depths 1 \
  --hy3-oq2e-model-root "$SEL" \
  --hy3-oq2e-manifest "$SEL/expert-manifest.json" \
  --hy3-oq2e-mtp-artifacts "$SEL/mtp" \
  --output-json "$out" \
  >> "$log" 2>&1
rc=$?
echo "[publish-rehearsal] === exit=$rc $(date -u +%FT%TZ) ===" >> "$log"
echo "[publish-rehearsal] exit=$rc" >&2
exit $rc
