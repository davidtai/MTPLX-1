# PR 391 battery, per-request rows, 2026-09-03

Terms used in this document:

- MTP: multi-token prediction.
- TTFT: time to first token, measured by the client.

This document holds one row per timed request of the cold battery. The rows
come from the `records[]` block of each receipt, not from the rendered report.
There are 63 timed cells: 21 per engine, with 3 seeds from the vanity cell to
65,536 tokens, 2 seeds at 131,072 tokens and 1 seed at 261,120 tokens.

Column sources:

- `Prefill s` is `prefill_time_s` and `Prefill tok/s` is `prefill_tok_s`, both as the server reported them.
- `TTFT s` is measured by the client.
- `Peak GB` is `peak_memory_gb` from the source named in `peak_memory_source`.

Every cell of every arm ran cold: `new_prefill_tokens` equals `prompt_tokens`
on all 63 rows. The branch arm reaches that state explicitly, through
`MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0` and
`MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS=999999999` in `cli_env_overrides`.
The upstream and mlx-serve receipts carry an empty `cli_env_overrides` and
restored nothing of their own accord. The quarantined receipt directories are
not included here; the warm-prefix run is in
`pr391-battery-2026-09-03-warm-prefix.md`.

---

## 1. upstream 2.10.2, control

| Cell | Seed | Prompt tok | New prefill tok | Prefill s | Prefill tok/s | TTFT s | Decode tok/s | Completion tok | Finish | Wall s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vanity | 20260829 | 87 | 87 | 0.271 | 321.6 | 0.318 | 85.82 | 201 | stop | 2.65 | 84.16 |
| vanity | 20260830 | 87 | 87 | 0.231 | 377.0 | 0.235 | 79.55 | 152 | stop | 2.16 | 84.17 |
| vanity | 20260831 | 87 | 87 | 0.238 | 366.2 | 0.243 | 81.40 | 254 | stop | 3.35 | 84.44 |
| 1K | 20260829 | 1,023 | 1,023 | 1.338 | 764.6 | 1.470 | 71.24 | 807 | stop | 12.79 | 85.19 |
| 1K | 20260830 | 1,024 | 1,024 | 1.138 | 900.0 | 1.279 | 66.43 | 1,024 | length | 16.68 | 85.63 |
| 1K | 20260831 | 1,024 | 1,024 | 1.138 | 900.0 | 1.443 | 66.20 | 1,024 | length | 16.90 | 86.08 |
| 8K | 20260829 | 8,192 | 8,192 | 7.057 | 1160.8 | 7.226 | 55.04 | 1,024 | length | 25.83 | 87.87 |
| 8K | 20260830 | 8,192 | 8,192 | 6.839 | 1197.9 | 7.023 | 59.16 | 809 | stop | 20.74 | 87.87 |
| 8K | 20260831 | 8,192 | 8,192 | 6.853 | 1195.5 | 7.114 | 52.34 | 704 | stop | 20.56 | 87.87 |
| 16K | 20260829 | 16,384 | 16,384 | 13.893 | 1179.3 | 14.088 | 58.72 | 1,024 | length | 31.53 | 89.57 |
| 16K | 20260830 | 16,384 | 16,384 | 13.676 | 1198.0 | 13.873 | 56.14 | 696 | stop | 26.29 | 89.57 |
| 16K | 20260831 | 16,384 | 16,384 | 13.699 | 1196.0 | 13.955 | 58.09 | 1,011 | stop | 31.38 | 89.57 |
| 32K | 20260829 | 32,768 | 32,768 | 30.570 | 1071.9 | 30.828 | 54.14 | 1,024 | length | 49.81 | 92.02 |
| 32K | 20260830 | 32,768 | 32,768 | 30.402 | 1077.8 | 30.667 | 57.86 | 1,024 | length | 48.38 | 92.02 |
| 32K | 20260831 | 32,768 | 32,768 | 30.338 | 1080.1 | 30.660 | 54.01 | 1,024 | length | 49.63 | 92.02 |
| 64K | 20260829 | 65,536 | 65,536 | 65.449 | 1001.3 | 65.829 | 62.26 | 672 | stop | 76.68 | 92.03 |
| 64K | 20260830 | 65,536 | 65,536 | 64.058 | 1023.1 | 64.455 | 52.75 | 559 | stop | 75.31 | 92.03 |
| 64K | 20260831 | 65,536 | 65,536 | 64.072 | 1022.9 | 64.526 | 54.79 | 557 | stop | 74.95 | 92.03 |
| 128K | 20260829 | 131,072 | 131,072 | 134.907 | 971.6 | 135.621 | 52.39 | 483 | stop | 145.79 | 91.58 |
| 128K | 20260830 | 131,072 | 131,072 | 132.966 | 985.8 | 133.556 | 49.95 | 671 | stop | 147.94 | 91.58 |
| 255K | 20260829 | 261,120 | 261,120 | 274.227 | 952.2 | 275.476 | 45.86 | 225 | stop | 280.66 | 96.33 |

---

## 2. branch, full stack

| Cell | Seed | Prompt tok | New prefill tok | Prefill s | Prefill tok/s | TTFT s | Decode tok/s | Completion tok | Finish | Wall s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vanity | 20260829 | 87 | 87 | 0.267 | 326.0 | 0.314 | 95.78 | 206 | stop | 2.49 | 84.63 |
| vanity | 20260830 | 87 | 87 | 0.229 | 380.5 | 0.234 | 104.11 | 178 | stop | 1.97 | 84.63 |
| vanity | 20260831 | 87 | 87 | 0.234 | 371.4 | 0.241 | 101.58 | 212 | stop | 2.32 | 84.76 |
| 1K | 20260829 | 1,023 | 1,023 | 1.320 | 775.1 | 1.451 | 82.27 | 979 | stop | 13.34 | 85.45 |
| 1K | 20260830 | 1,024 | 1,024 | 1.110 | 922.3 | 1.228 | 82.48 | 1,024 | length | 13.64 | 85.73 |
| 1K | 20260831 | 1,024 | 1,024 | 1.093 | 937.2 | 1.278 | 93.65 | 793 | stop | 9.74 | 87.06 |
| 8K | 20260829 | 8,192 | 8,192 | 7.695 | 1064.5 | 7.866 | 86.17 | 1,024 | length | 19.75 | 91.15 |
| 8K | 20260830 | 8,192 | 8,192 | 6.772 | 1209.7 | 6.922 | 83.81 | 1,024 | length | 19.14 | 93.99 |
| 8K | 20260831 | 8,192 | 8,192 | 6.770 | 1210.0 | 6.981 | 78.87 | 562 | stop | 14.11 | 95.44 |
| 16K | 20260829 | 16,384 | 16,384 | 12.633 | 1296.9 | 12.826 | 81.88 | 1,024 | length | 25.34 | 97.64 |
| 16K | 20260830 | 16,384 | 16,384 | 12.158 | 1347.6 | 12.325 | 77.13 | 610 | stop | 20.25 | 98.86 |
| 16K | 20260831 | 16,384 | 16,384 | 12.133 | 1350.3 | 12.335 | 83.75 | 990 | stop | 24.18 | 99.37 |
| 32K | 20260829 | 32,768 | 32,768 | 25.521 | 1284.0 | 25.762 | 78.58 | 1,024 | length | 38.87 | 101.75 |
| 32K | 20260830 | 32,768 | 32,768 | 25.201 | 1300.3 | 25.415 | 78.06 | 855 | stop | 36.44 | 103.94 |
| 32K | 20260831 | 32,768 | 32,768 | 25.193 | 1300.7 | 25.449 | 74.96 | 442 | stop | 31.41 | 103.94 |
| 64K | 20260829 | 65,536 | 65,536 | 52.055 | 1259.0 | 52.398 | 69.23 | 1,024 | length | 67.42 | 103.94 |
| 64K | 20260830 | 65,536 | 65,536 | 51.871 | 1263.4 | 52.211 | 71.34 | 435 | stop | 58.53 | 103.94 |
| 64K | 20260831 | 65,536 | 65,536 | 51.860 | 1263.7 | 52.189 | 79.93 | 1,024 | length | 65.23 | 103.94 |
| 128K | 20260829 | 131,072 | 131,072 | 107.887 | 1214.9 | 108.449 | 76.11 | 1,024 | length | 122.88 | 97.21 |
| 128K | 20260830 | 131,072 | 131,072 | 107.800 | 1215.9 | 108.354 | 70.42 | 592 | stop | 117.75 | 97.22 |
| 255K | 20260829 | 261,120 | 261,120 | 219.629 | 1188.9 | 220.629 | 65.53 | 1,020 | stop | 238.97 | 104.22 |

---

## 3. mlx-serve 26.8.11

| Cell | Seed | Prompt tok | New prefill tok | Prefill s | Prefill tok/s | TTFT s | Decode tok/s | Completion tok | Finish | Wall s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vanity | 20260829 | 87 | 87 | 0.219 | 397.5 | 0.244 | 92.65 | 210 | stop | 2.51 | 73.25 |
| vanity | 20260830 | 87 | 87 | 0.211 | 412.1 | 0.235 | 89.85 | 200 | stop | 2.47 | 73.25 |
| vanity | 20260831 | 87 | 87 | 0.211 | 411.6 | 0.235 | 99.88 | 181 | stop | 2.04 | 73.26 |
| 1K | 20260829 | 1,023 | 1,023 | 0.652 | 1569.7 | 0.677 | 74.75 | 453 | stop | 6.75 | 74.27 |
| 1K | 20260830 | 1,024 | 1,024 | 0.619 | 1654.4 | 0.644 | 64.75 | 991 | stop | 15.93 | 74.27 |
| 1K | 20260831 | 1,024 | 1,024 | 0.617 | 1659.6 | 0.642 | 62.80 | 396 | stop | 6.95 | 74.27 |
| 8K | 20260829 | 8,192 | 8,192 | 4.616 | 1774.6 | 4.649 | 61.28 | 1,024 | length | 21.34 | 77.37 |
| 8K | 20260830 | 8,192 | 8,192 | 4.503 | 1819.3 | 4.535 | 61.22 | 795 | stop | 17.50 | 77.37 |
| 8K | 20260831 | 8,192 | 8,192 | 4.501 | 1820.0 | 4.533 | 62.20 | 1,024 | length | 20.98 | 77.37 |
| 16K | 20260829 | 16,384 | 16,384 | 10.930 | 1499.0 | 10.969 | 68.89 | 986 | stop | 25.29 | 77.80 |
| 16K | 20260830 | 16,384 | 16,384 | 10.850 | 1510.1 | 10.889 | 64.91 | 956 | stop | 25.62 | 77.80 |
| 16K | 20260831 | 16,384 | 16,384 | 10.846 | 1510.6 | 10.885 | 63.40 | 806 | stop | 23.60 | 77.80 |
| 32K | 20260829 | 32,768 | 32,768 | 28.739 | 1140.2 | 28.781 | 58.73 | 595 | stop | 38.92 | 79.28 |
| 32K | 20260830 | 32,768 | 32,768 | 28.764 | 1139.2 | 28.807 | 63.58 | 799 | stop | 41.36 | 79.28 |
| 32K | 20260831 | 32,768 | 32,768 | 28.758 | 1139.4 | 28.800 | 58.98 | 1,024 | length | 46.14 | 79.28 |
| 64K | 20260829 | 65,536 | 65,536 | 81.337 | 805.7 | 81.395 | 50.33 | 832 | stop | 97.92 | 82.11 |
| 64K | 20260830 | 65,536 | 65,536 | 81.381 | 805.3 | 81.440 | 53.28 | 632 | stop | 93.31 | 82.12 |
| 64K | 20260831 | 65,536 | 65,536 | 81.559 | 803.5 | 81.620 | 53.12 | 1,024 | length | 100.88 | 82.12 |
| 128K | 20260829 | 131,072 | 131,072 | 229.399 | 571.4 | 229.493 | 43.24 | 1,024 | length | 253.15 | 88.07 |
| 128K | 20260830 | 131,072 | 131,072 | 229.078 | 572.2 | 229.169 | 48.29 | 1,024 | length | 250.35 | 88.07 |
| 255K | 20260829 | 261,120 | 261,120 | 597.501 | 437.0 | 597.681 | 35.15 | 1,024 | length | 626.78 | 99.54 |

---

## 4. Engine version block, per engine

The harness records this block at launch, one block per receipt file. The
block is identical across the three receipt files of each arm. `git_head` is
the head of the tree that each server imported, not the branch tip of the pull
request.

| Engine | Receipt files | source | engine | mlx | mlx_c | git_head |
| --- | --- | --- | --- | --- | --- | --- |
| `upstream-2.10.2` | `upstream-2.10.2-1788410820.json`<br>`upstream-2.10.2-1788413632.json`<br>`upstream-2.10.2-1788415574.json` | `importlib.metadata:mtplx` | **2.10.2** | 0.32.2 | n/a | `8bc4d88e421d68a9b9e29edc75ff14ed3b81bfef` |
| `branch-fullstack` | `branch-fullstack-1788418096.json`<br>`branch-fullstack-1788419204.json`<br>`branch-fullstack-1788419696.json` | `importlib.metadata:mtplx` | **2.10.1** | 0.32.2 | n/a | `5c62e89d9fe6d2dcbc77801a00ba208c045f3740` |
| `mlx-serve` | `mlx-serve-1788412531.json`<br>`mlx-serve-1788414592.json`<br>`mlx-serve-1788415978.json` | `mlx-serve --version` | **26.8.11** | 0.32.2 | `56b2d39fc831` | `n/a` |

Raw `engine_version.raw` strings, verbatim:

```
upstream-2.10.2:
  mtplx 2.10.2
  mlx 0.32.2
branch-fullstack:
  mtplx 2.10.1
  mlx 0.32.2
mlx-serve:
  mlx-serve 26.8.11
  mlx 0.32.2
  mlx-c 56b2d39fc831
  nax on (M5 neural accelerators)
  ggml 0.20.1 (60eeeb608)
  llama.cpp b10472
  gguf 3
  ds4 unknown
```

`engine_version_reason` is null on all 9 timed-run receipt files. Every probe
succeeded, so no arm falls back to the hand-maintained string of the renderer.

---

## 5. Request body digest, per cell

All three engines were asked for the same thing on every cell. 21 of 21 cell
and seed pairs ran on more than one engine. All 21 carry a byte-identical
`request_body_sha256` across the three engines, and 0 pairs differ. The digest
covers every request body field except `model`.

| Cell | Seed | request_body_sha256 (identical on all 3 engines) |
| --- | ---: | --- |
| vanity | 20260829 | `a42cd1742170b1dd8ac5567d5fa57d8003e25784bdb0462126e82328c5105dca` |
| vanity | 20260830 | `4a316867cf5254d6e236cd5509f660b4fafdd826e2029f9ed5337c91c911895a` |
| vanity | 20260831 | `754f9a191062b695f02e9c7358f87181c2db34ceb7b851e31b72e9aed18d5100` |
| 1K | 20260829 | `bdd51de8b6bf8f5e727c256fcf98a1a3e25b2e5783401d5c0c8c98d033b512af` |
| 1K | 20260830 | `0d5fcf56046def2b1f1e2957c6aca5c62ebede5587ddeda1f24736eee2b2fb56` |
| 1K | 20260831 | `4c4a69d00bac24fd999f93d451a1d1db9255189baedf2dab59f7d321fa127b62` |
| 8K | 20260829 | `caeb8d82384b71abe6e56bcda47384d46ac05eee6fb7cf899f5da9795c3b4614` |
| 8K | 20260830 | `5b0e6aea099da4ca5048cc37e61dd0d108819aa79d331a8c64536f148ae2e5b9` |
| 8K | 20260831 | `1d40e873d37386ad70d73568dcf24714f231053b87cb2b380acbd7b62bb14ab7` |
| 16K | 20260829 | `eb463f6ca7bb8f8a08a300b96d1c5b5783ced23f25a17327a5232ab7193f6e2d` |
| 16K | 20260830 | `fd0e34915036f16c840b87f4e52b62fedec9e2fd785ec565f7da96edced3ad44` |
| 16K | 20260831 | `47e7cd7822afab0e2f0f834c150a2b3e3cd5600400e0adb9dc3c56094bd3b8e3` |
| 32K | 20260829 | `0d952d439c398f638c17d1cba1440360c350cabffe31679e8b84d32e9740c99a` |
| 32K | 20260830 | `869fa599a28a6edeb50d27623ff411aafe2ef35dd57ae21d56c495e101f6e263` |
| 32K | 20260831 | `f795aa6a38ec2f1e30accd8717058dd767309227268658e6fbfd2158d581f1bd` |
| 64K | 20260829 | `6707e739f1ac5e5f666afa34f015849c60e1f8029b59fbe15e9ad63d319c30e4` |
| 64K | 20260830 | `8cb00825429de68a14fd53185afab286f1ba7212f5084663a6bf6f865409f896` |
| 64K | 20260831 | `72c73c2b993e9b60f253760bacfaddf2311a0b88d95d6c00c7e0d00d03712c5b` |
| 128K | 20260829 | `7dba466982846861dab64fcda296110370b62fa4325d54f125204d825388b8ea` |
| 128K | 20260830 | `ed7c6a78bd63aa79a5b956f6763bceb0ff5ea30b9418900f946013fc30a9f613` |
| 255K | 20260829 | `dffea3fc9673eedace109054742f190062e479a6edced5d841ac6759ab0ce538` |

Request body as launched, from the vanity cell of every arm. `model` is the
only field that differs, by construction.

```json
{
  "enable_thinking": false,
  "max_tokens": 1024,
  "model": "<per-engine served model id>",
  "seed": 20260829,
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95
}
```

---

## 6. Receipt fields not found

None. Every one of the 13 requested fields is present and not null on all 63
timed records: `cell` or `target_tokens`, `seed`, `prompt_tokens`,
`new_prefill_tokens`, `prefill_time_s`, `prefill_tok_s`, `ttft_s`,
`decode_tok_s`, `completion_tokens`, `finish_reason`, `wall_s`,
`peak_memory_gb` and `request_body_sha256`.

Two notes on how the columns are derived:

- The receipts label only two `cell` values, `vanity` and `sweep`. The size column is `target_tokens` put through `cell_label()` in `scripts/fable/server_cell_report.py`, which renders `261120` as `255K`.
- `peak_footprint_gb`, `mtp_accept_rate`, `thermal_gate`, `fans`, `power`, `residency`, `vm_stat` and `raw_engagement` are on every record. The report summarises them in its run-conditions and engagement tables instead of repeating them per row.
