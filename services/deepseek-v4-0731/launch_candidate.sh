#!/bin/sh
# Isolated candidate only. This file never manages the production service.
set -eu
umask 077

SERVICE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/services/deepseek-v4-0731
WORKTREE=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp
PYTHON=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv/bin/python
PYTHON_TARGET=/Users/davidtai/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
ENTRY="$SERVICE_ROOT/candidate_entry.py"
ENCODING="$SERVICE_ROOT/encoding"
REVIEWED_REF=refs/tags/mtplx-dsv4-0731-reviewed
PORT=8081

die() { printf '%s\n' "deepseek-v4-0731 candidate: $1" >&2; exit 64; }
sha256() { /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }

fixture=${MTPLX_DSV4_0731_TEST_FIXTURE:-}
if [ -n "${MTPLX_DSV4_0731_EXECUTABLE:-}" ] && [ "$fixture" != 1 ]; then
  die "command environment override rejected"
fi
if [ "$fixture" = 1 ]; then
  [ "${1:-}" = --print-command ] || die "fixture mode only permits --print-command"
  printf '%s\n' "$PYTHON $ENTRY (fixed 127.0.0.1:$PORT)"
  exit 0
fi
[ "$#" -eq 0 ] || die "arguments are not accepted"

[ -L "$PYTHON" ] && [ "$(/usr/bin/readlink "$PYTHON")" = "$PYTHON_TARGET" ] || die "trusted python link changed"
[ -x "$PYTHON_TARGET" ] && [ ! -L "$PYTHON_TARGET" ] || die "trusted python target is missing or unsafe"
[ "$(sha256 "$PYTHON_TARGET")" = 96793b100c947cdc81a38e8fb8c9c1889abccda9840ce1bef58d372bf3f2c263 ] || die "trusted python hash changed"
[ -f "$ENTRY" ] && [ ! -L "$ENTRY" ] || die "candidate entrypoint is missing or unsafe"
[ "$(sha256 "$ENTRY")" = c7e2e79e45b3f2e8afd8453d5976e5234caea724813eae0e555c5f956bf725aa ] || die "candidate entrypoint hash changed"
[ -d "$MODEL" ] && [ ! -L "$MODEL" ] || die "pinned model path is missing or unsafe"
[ "$(sha256 "$MODEL/config.json")" = c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f ] || die "model configuration hash changed"
[ "$(sha256 "$MODEL/model.safetensors.index.json")" = c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8 ] || die "model index hash changed"
[ "$(sha256 "$ENCODING/SHA256SUMS")" = 6758dfda8a39afdd00d907606c42c1a268289c463351b9628ac07f4f916d7d0a ] || die "official encoding manifest hash changed"
(cd "$ENCODING" && /usr/bin/shasum -a 256 -c SHA256SUMS >/dev/null) || die "official encoding/vector asset hash changed"

reviewed_commit=$(/usr/bin/git -C "$WORKTREE" rev-parse --verify "${REVIEWED_REF}^{commit}") || die "reviewed commit ref is missing"
current_commit=$(/usr/bin/git -C "$WORKTREE" rev-parse --verify HEAD) || die "worktree HEAD is missing"
[ "$current_commit" = "$reviewed_commit" ] || die "worktree is not the exact reviewed commit"
[ -z "$(/usr/bin/git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all)" ] || die "reviewed worktree is not clean"

exec /usr/bin/env -i \
  HOME=/Users/davidtai \
  LC_ALL=C \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$WORKTREE" \
  VIRTUAL_ENV=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv \
  "$PYTHON" "$ENTRY"
