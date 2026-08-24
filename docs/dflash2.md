# Qwen3.8 DFlash2

MTPLX supports Qwen3.8 DFlash2 as the explicit `dflash2` backend. The bundle
keeps the Qwen target and the DFlash2 draft in separate directories; it is not
a native `mtp.safetensors` sidecar.

## Install and launch

Install the pinned optional dependency:

```sh
.venv/bin/python -m pip install 'mtplx[dflash2]'
# The extra provides the pinned dflash-mlx runtime used by the 54-row campaign.
```

A bundle contains `mtplx_dflash2.json`, `target/`, and `dflash2/`. A normal
launch auto-detects the bundle; `--backend-id dflash2` makes the selection
explicit:

```sh
mtplx serve --model /models/qwen38-dflash2 --backend-id dflash2
mtplx quickstart --model /models/qwen38-dflash2 --backend-id dflash2
mtplx ask --model /models/qwen38-dflash2 --backend-id dflash2 "Explain speculative decoding."
```

These commands detect the DFlash manifest before importing MLX and latch the
measured row-53 command-buffer environment. Direct Python embedding must set
`MLX_MAX_MB_PER_BUFFER=512` and `MLX_MAX_OPS_PER_BUFFER=50` before importing
`mtplx` or MLX; the loader fails closed if that process contract is absent.

DFlash2 defaults are sampler `temperature=1.0`, `top_p=0.95`, `top_k=20`, and
physical block `8`. The retained adaptive row-15 policy selects physical blocks
from `1` through `8`; `--depth` or `--draft-block-size` changes the ceiling.
`--generation-mode ar` or `--no-mtp` deliberately routes to the bundle's
`target/` model with MTP disabled; it never loads the DFlash2 draft or silently
falls back to native MTP.

DFlash generation uses the server's serial model-owner lane. Request-specific
target sampling is protected by one process-wide generation lock because the
pinned DFlash engine exposes a process-global sampler seam; direct concurrent
thread callers therefore serialize. The benchmark and production receipt make
no concurrent-throughput claim.

The portable Homebrew binary path is:

```sh
$(brew --prefix mtplx)/bin/mtplx serve --model /models/qwen38-dflash2
```

For another installation, set `MTPLX_BREW_VENV` to a virtual-environment
directory or Python executable. It contains no credentials:

```sh
MTPLX_BREW_VENV=/path/to/venv mtplx serve --model /models/qwen38-dflash2
```

## Manifest contract

The manifest pins both revisions, records draft precision, and records SHA-256
checksums. Paths are relative to the bundle root:

```json
{
  "schemaVersion": 1,
  "backend": "dflash2",
  "target": {
    "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
    "base_model": "Qwen/Qwen3.8-27B",
    "revision": "123db8bcc7101455b00d9aad36c0e760c6e7de02",
    "precision": "unquantized"
  },
  "draft": {
    "repo": "z-lab/Qwen3.8-27B-DFlash2",
    "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
    "precision": "4bit"
  },
  "layout": {"target": "target", "draft": "dflash2"},
  "checksums": {
    "target_config": {
      "path": "target/config.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "target_weights": [
      {
        "path": "target/model-00001-of-N.safetensors",
        "sha256": "<64 lowercase hex characters>"
      }
    ],
    "draft_config": {
      "path": "dflash2/config.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "draft_weights": [
      {
        "path": "dflash2/model.safetensors",
        "sha256": "<64 lowercase hex characters>"
      }
    ]
  },
  "algorithm": {
    "repo": "davidtai/dflash-mlx",
    "revision": "c5b76ddb62bdefb6eeef1282641842edcf23a1b8",
    "version": "0.1.10"
  }
}
```

The canonical resolver fails closed on an invalid manifest. It verifies both
the manifest checksums and MTPLX's independent hard-coded inventory/digests for
the exact measured target config, six target weight files, draft config, and
draft weight file. Rewriting the manifest cannot authorize substituted bytes.
Do not rename DFlash2 tensors to `mtp.*` or modify source checkpoints.

## Comparable benchmarks

Use one target, prompt set, sampler, and token budget for every run. The
official single-prompt parity command is:

```sh
TARGET=/models/Qwen3.8-27B
DRAFT=/models/Qwen3.8-27B-DFlash2
PROMPT='Explain speculative decoding in one paragraph.'
TOKENS=128
TEMP=1.0
TOP_P=0.95
TOP_K=20

# Pinned dflash-mlx single-prompt greedy parity.
dflash generate --model "$TARGET" --draft "$DRAFT" \
  --draft-quant w4:gs64 --verify-mode dflash --copyspec-mode off \
  --verify-len-cap 8 --max-tokens "$TOKENS" --prompt "$PROMPT"
```

For same-harness performance over a prompt file, run DFlash and native MTPLX
with the same target, prompt file, sampler, and token budget:

```sh
BUNDLE=/models/qwen38-dflash2
PROMPTS=/path/to/prompts.jsonl
OUTPUT=/tmp/mtplx-dflash-baseline.jsonl

mtplx dflash-mlx-baseline --model "$TARGET" --draft-model "$DRAFT" \
  --prompts "$PROMPTS" --temperature "$TEMP" --top-p "$TOP_P" \
  --top-k "$TOP_K" --max-tokens "$TOKENS" --block-size 8 \
  --output "$OUTPUT"

mtplx mtp-depth-sweep --model "$TARGET" --prompts "$PROMPTS" \
  --depths 1,2,3,4,5,6,7,8 --compare-ar --temperature "$TEMP" \
  --top-p "$TOP_P" --top-k "$TOP_K" --max-tokens "$TOKENS"
```

The sweep's `--compare-ar` is the native MTPLX target-only AR comparison.
Record target and draft revisions, acceptance length, generated-token
throughput, and peak memory. Compare unquantized, 8-bit, and 4-bit drafts only
after deterministic committed tokens match target-only AR. MTPLX does not use
llama.cpp for this integration.

Known-risk upstream parity references: [z-lab/dflash#159](https://github.com/z-lab/dflash/issues/159)
and [z-lab/dflash#160](https://github.com/z-lab/dflash/issues/160).
