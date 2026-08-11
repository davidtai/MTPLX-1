#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR="${TMPDIR:-/tmp}"
WORKDIR="$(mktemp -d "$TMPDIR/mtplx-fresh-venv.XXXXXX")"
VENV="$WORKDIR/.venv"
MODEL_DIR="$WORKDIR/non-mtp-model"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
# Keep MLX absent while providing the non-MLX dependency imported by the
# expert-inspection path exercised below.
"$VENV/bin/python" -m pip install "numpy>=2" >/dev/null

shopt -s nullglob
wheels=("$ROOT"/dist/*.whl)
shopt -u nullglob

if [[ "${#wheels[@]}" -eq 0 ]]; then
  echo "fresh_venv_smoke: no wheel found in $ROOT/dist" >&2
  echo "Run: python -m build" >&2
  exit 2
fi

"$VENV/bin/python" -m pip install --no-deps "${wheels[0]}" >/dev/null

mkdir -p "$MODEL_DIR"
printf '{"model_type":"llama"}\n' > "$MODEL_DIR/config.json"

"$VENV/bin/mtplx" --help >/dev/null
"$VENV/bin/mtplx" doctor --json >/dev/null
set +e
"$VENV/bin/mtplx" inspect "$MODEL_DIR" --json >/dev/null
INSPECT_STATUS=$?
set -e
if [ "$INSPECT_STATUS" -ne 2 ]; then
  echo "expected no-MTP inspect to exit 2, got $INSPECT_STATUS" >&2
  exit 1
fi
"$VENV/bin/mtplx" init --dry-run --json --config "$WORKDIR/config.toml" >/dev/null

echo "fresh_venv_smoke: passed no-MLX CLI survival checks"
