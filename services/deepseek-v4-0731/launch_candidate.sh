#!/bin/sh
# Isolated candidate only.  This file must never manage the production service.
set -eu
umask 077

SERVICE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/services/deepseek-v4-0731
WORKTREE=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp
PYTHON=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv/bin/python
PYTHON_TARGET=/Users/davidtai/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
CONFIG="$SERVICE_ROOT/candidate.json"
ASSET="$SERVICE_ROOT/encoding/chat_template.jinja"
MANIFEST="$SERVICE_ROOT/encoding/SHA256SUMS"

die() { printf '%s\n' "deepseek-v4-0731 candidate: $1" >&2; exit 64; }
sha256() { /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }

# Command overrides are only possible in an explicit, local test fixture.  A
# launchd job never sets this flag, and production accepts no caller argv/env.
fixture=${MTPLX_DSV4_0731_TEST_FIXTURE:-}
if [ -n "${MTPLX_DSV4_0731_EXECUTABLE:-}" ] && [ "$fixture" != 1 ]; then
  die "command environment override rejected"
fi
if [ "$fixture" = 1 ]; then
  [ "${1:-}" = --print-command ] || die "fixture mode only permits --print-command"
  printf '%s\n' "$PYTHON -m mtplx serve --host 127.0.0.1 --port 8081"
  exit 0
fi
[ "$#" -eq 0 ] || die "arguments are not accepted"

[ -r "$CONFIG" ] && [ ! -L "$CONFIG" ] || die "candidate configuration is missing or unsafe"
[ -L "$PYTHON" ] && [ "$(/usr/bin/readlink "$PYTHON")" = "$PYTHON_TARGET" ] || die "trusted python link changed"
[ -x "$PYTHON_TARGET" ] && [ ! -L "$PYTHON_TARGET" ] || die "trusted python target is missing or unsafe"
[ -d "$MODEL" ] && [ ! -L "$MODEL" ] || die "pinned model path is missing or unsafe"
[ -f "$ASSET" ] && [ ! -L "$ASSET" ] || die "encoding asset is missing or unsafe"
[ -f "$MANIFEST" ] && [ ! -L "$MANIFEST" ] || die "encoding manifest is missing or unsafe"

[ "$(sha256 "$ASSET")" = 03f2686beff14c3d9040894a2b658d9f1917be90bc1d90597502fc2562f0ec2a ] || die "encoding asset hash changed"
[ "$(sha256 "$PYTHON_TARGET")" = 96793b100c947cdc81a38e8fb8c9c1889abccda9840ce1bef58d372bf3f2c263 ] || die "trusted python hash changed"
[ "$(sha256 "$MODEL/config.json")" = c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f ] || die "model configuration hash changed"
[ "$(sha256 "$MODEL/model.safetensors.index.json")" = c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8 ] || die "model index hash changed"
[ "$(/usr/bin/git -C "$WORKTREE" merge-base HEAD 5ccc9fdf251a9eaf946f4c77c42eabd6ba3f0ab4)" = 5ccc9fdf251a9eaf946f4c77c42eabd6ba3f0ab4 ] || die "worktree does not descend from pinned revision"
/usr/bin/grep -Fqx '03f2686beff14c3d9040894a2b658d9f1917be90bc1d90597502fc2562f0ec2a  chat_template.jinja' "$MANIFEST" || die "encoding manifest changed"

# `env -i` is the process boundary: no caller environment reaches model load.
exec /usr/bin/env -i \
  HOME=/Users/davidtai \
  LC_ALL=C \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  VIRTUAL_ENV=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv \
  "$PYTHON" -m mtplx serve \
  --host 127.0.0.1 \
  --port 8081 \
  --model "$MODEL" \
  --model-id deepseek-v4-0731-candidate \
  --reasoning-effort low \
  --reasoning-parser none \
  --warmup-tokens 0 \
  --no-stats-footer
