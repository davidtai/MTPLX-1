# Qwen 3.8 Challenge Port Ledger

This ledger records the MTPLX disposition of all 82 promoted Yukon Qwen 3.8
submissions. The immutable source snapshot is
`receipts/qwen38-challenge-port/yukon-accepted-2026-08-23.json`; the row-by-row
mechanism mapping is in
`../../specs/2026-08-23-qwen38-challenge-port-design.md`.

## Source and benchmark contract

- Source gate: `(score / previous promoted score - 1) * 100 > 0.10`.
- Promoted Yukon rows: 82; above-threshold source rows: 54.
- Challenge pin: `eb5eadc7a165047d4321ce883b9ff30894d8bd19`.
- MTPLX base pin: `bd4421567f9e16ce957c6ef97708b072dcd73937`.
- Inventory check: `python scripts/qwen38_challenge_inventory.py --check`.

The 54 rows are not 54 independent ports. Controlled resamples, no-ops,
Swift/Metal worker plumbing, already-present MTPLX behavior, intermediate
variants superseded by later rows, and mechanisms later removed from the
accepted source sequence are skipped. Only the final transferable descendant
of each mechanism is implemented and measured.

Performance decisions are cumulative and chronological. Candidate `N` is
measured against the complete stack retained before `N`; a winner becomes the
next control. Percentages are measured matched wall-time changes, never added
or multiplied from isolated results.

The final gate uses exactly 16,384 Python input tokens and 1,024 generated
tokens. Each route receives one full conditioning generation, followed by four
timed ABBA arms. Target and draft sampling use temperature 1.0, top-p 0.95,
top-k 20, and seed 42. The runtime is the exact Optimized-Speed artifact with
Turbo, the installed Q4/group-64 draft head, depth 3, persistent committed
history, capture/commit verification, and `linear-gdn-from-conv-tape`. The
prompt is built from `mtplx/generation.py` with the intact
`python_modules_long.jsonl` instruction at the tail.

## Chronological decisions

| Order | Source rows / final descendant | Candidate | 16K wall delta | Decision |
| ---: | --- | --- | ---: | --- |
| 1 | 11, 20 | K/V-only committed-history append | **+1.9268%** | **Retain as S1**, active only when original request context is at least 16,384 tokens. Exact output/schedule; flat memory. |
| 2 | 21 | fused Q/K RMSNorm + partial RoPE | **-0.2034%** | Reject and remove. Token drift was not the rejection reason; wall time lost. |
| 3 | 45 | boundary residual/RMSNorm fusion | **-0.0130%** | Reject and remove as flat/losing; peak memory increased. |
| 4 | 60, 61 | dual pre-FC RMSNorm / concat-free output | **+0.0505%** | Reject and remove as below the 0.10% floor after a conditioned rerun on corrected S1. |
| 5 | 19, 34, 36, 39, 40, 41, 70, 78, 80 | final cross-row affine-4 QMV, adapted to group-32 trunk plus group-64 islands | **+0.0618%** | Reject and remove as below the 0.10% floor; decode throughput also slightly regressed. |
| 6 | 10, 42, 46, 47, 67, 69, 71, 79, 82 | final compact Q2 coarse/exact proposal head | **-1.3792%** | Reject and remove. |

The final production stack is therefore **S1: K/V-only history at request
context >=16,384**. The route uses the original request length rather than the
truncated 8K committed-history window. Below the threshold the runtime calls
the existing stock append directly.

## 16K prefill/decode/memory table

Means are over two timed arms per route; peak memory is the maximum arm value.
Wall delta is computed from mean total wall time. Absolute throughput varies
between brackets, so only the matched routes within one receipt are compared.

| Candidate | Arm | Prefill tok/s | Decode tok/s | Peak GiB | Mean wall s | Generated | Wall delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K/V-only history | C: Optimized-Speed | 746.426 | 52.224 | 24.885 | 42.058 | 1,024 | - |
|  | N: K/V-only >=16K | 771.941 | 51.397 | 24.885 | 41.263 | 1,024 | **+1.9268%** |
| Q/K norm + RoPE | C: S1 | 774.028 | 51.919 | 24.885 | 40.990 | 1,024 | - |
|  | N: S1 + Q/K | 776.849 | 51.492 | 24.885 | 41.073 | 1,024 | **-0.2034%** |
| Boundary norm | C: S1 | 781.277 | 51.681 | 24.885 | 40.885 | 1,024 | - |
|  | N: S1 + boundary | 782.377 | 51.616 | 24.944 | 40.891 | 1,024 | **-0.0130%** |
| Dual norm | C: S1 | 782.799 | 52.099 | 24.885 | 40.682 | 1,024 | - |
|  | N: S1 + dual | 786.854 | 51.911 | 24.885 | 40.661 | 1,024 | **+0.0505%** |
| Final QMV | C: S1 | 782.860 | 51.626 | 24.885 | 40.861 | 1,024 | - |
|  | N: S1 + QMV | 784.808 | 51.592 | 24.885 | 40.836 | 1,024 | **+0.0618%** |
| Compact head | C: S1 | 760.692 | 50.715 | 24.885 | 41.841 | 1,024 | - |
|  | N: S1 + compact | 759.563 | 49.345 | 24.885 | 42.426 | 1,024 | **-1.3792%** |

K/V has direct causal evidence beyond the aggregate wall delta. Its two timed
arms reduced prompt MTP-history append from 0.448-0.485 seconds to
0.112-0.116 seconds. Dual norm's corrected rerun engaged 877 calls in both
candidate arms but improved total wall by only 0.0505%, below the requested
floor.

The Q/K receipt predates the final tie-breaking classification and marks its
cross-route token drift as `correctness.passed=false`. Both routes were
individually deterministic, produced all 1,024 tokens, and had identical depth
schedules. Its rejection is the measured 0.2034% wall regression, not the token
hash difference.

## Skipped source mechanisms

| Classification | Mechanisms | Reason |
| --- | --- | --- |
| Already present | fused target Q/K/V projection; GDN `in_proj_qkvz` / `in_proj_ba`; verify-hidden reuse; compiled attention gate; committed-history and replay infrastructure; Q4 draft head; warmup and EV/cost policy | Optimized-Speed already owns the same work or a broader compiled boundary. A duplicate port is a no-op or adds dispatch overhead. |
| Weak / no-op | 27 rows at or below 0.10%, controlled resamples, bookkeeping-only changes | No qualifying source performance claim. |
| Superseded / removed later | intermediate compact-head, QMV, fusion, and calibration variants | Only the final surviving descendant receives an MTPLX gate. |
| Challenge-only | Swift worker plumbing, declared-head staging, Metal command-buffer/residency policy, trusted-worker target top-2 ledger | No equivalent consumer or ownership boundary exists in Python MTPLX. |
| Source-specific | one-forward SDPA workaround and source depth floor 6/cap 7 | MTPLX has no matching SDPA width wall and the measured policy is depth 3. |
| Artifact/provenance | challenge precision-island compact artifact | It targets a different head, lacks redistribution metadata, and its final compact descendant loses. |

## Authoritative receipts

- Retained S1:
  `conditioned-s1-kv-ge16384-request-context-python-long16384in-1024out-t1-abba-2026-08-23.json`.
- Corrected dual-on-S1 rejection:
  `conditioned-r61-dual-stack-corrected-s1-python-long16384in-1024out-t1-abba-2026-08-23.json`.
- Remaining chronological rejections: the `chrono-r21`, `chrono-r45`,
  `chrono-r80`, and `chrono-r82` 16K receipts in the same directory.
- QMV numerical tie audit:
  `qmv-final-g32-g64-real-model-numeric-parity-2026-08-23.json`. QMV was
  rejected for performance, not parity.
- Static benchmark contract:
  `receipts/qwen38-challenge-port/control-contract-2026-08-23.md`.

The final route receipt names only
`qwen38_mtp_kv_only_history_ge16384_v1`; rejected experimental kernels are
absent from the production path.
