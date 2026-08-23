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
- MTPLX branch base pin: `c1300f17c66e5fef7810efa80f2a53489b8d001d`.
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
| 1 | 3 | packed target attention Q/K/V | **-4.0629%** | Reject and remove. The corrected isolated arm engaged attention only; the earlier positive arm was invalid because it also packed MLP projections. |
| 2 | 13 | fused GDN QKV+Z and B+A projection pairs | **-7.2989%** | Reject and remove after 7,392 engaged pair calls per candidate arm. |
| 3 | 11, 20 | K/V-only committed-history append | **+1.9268%** | **Retain as S1**, active only when original request context is at least 16,384 tokens. Exact output/schedule; flat memory. |
| 4 | 21 | fused Q/K RMSNorm + partial RoPE | **-0.2034%** | Reject and remove. Token drift was not the rejection reason; wall time lost. |
| 5 | 45 | boundary residual/RMSNorm fusion | **-0.0130%** | Reject and remove as flat/losing; peak memory increased. |
| 6 | 60, 61 | dual pre-FC RMSNorm / concat-free output | **+0.0505%** | **Retain as S2** under the revised strict `>0.05%` local gate. |
| 7 | 19, 34, 36, 39, 40, 41, 70, 78, 80 | final cross-row affine-4 QMV, adapted to group-32 target trunk and group-64 source head | **+0.2528%** | **Retain as S3** after correcting the group-size indexing and measuring on S1+S2. |
| 8 | 10, 42, 46, 47, 67, 69, 71, 79, 82 | source Q4 proposal body with BF16 Q/K/V islands, E87 Q2 cluster shortlist, fused row top-32, and selected Q4 rerank | **+2.8631%** | **Retain as S4**. Both routes were deterministic and completed 1,024 tokens; proposal drift is allowed. |

The final production stack is **S4: S1 K/V-only history + S2 dual norm + S3
target-shaped final QMV + S4 source proposal stack**. The history route uses
the original request length rather than the truncated 8K committed-history
window. Below the threshold it falls back to the existing stock append. The
proposal artifact is immutable and proposal-only; every emitted token remains
target-verified.

## 16K prefill/decode/memory table

Means are over two timed arms per route; peak memory is the maximum arm value.
Wall delta is computed from mean total wall time. Absolute throughput varies
between brackets, so only the matched routes within one receipt are compared.

| Candidate | Arm | Prefill tok/s | Decode tok/s | Peak GiB | Mean wall s | Generated | Wall delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed Q/K/V | C: Optimized-Speed | 754.031 | 51.803 | 25.569 | 42.006 | 1,024 | - |
|  | N: packed attention Q/K/V | 761.534 | 46.971 | 25.569 | 43.785 | 1,024 | **-4.0629%** |
| GDN pairs | C: Optimized-Speed | 756.716 | 52.217 | 27.244 | 41.744 | 1,024 | - |
|  | N: fused QKVZ + BA | 694.867 | 49.341 | 27.244 | 45.031 | 1,024 | **-7.2989%** |
| K/V-only history | C: Optimized-Speed | 746.426 | 52.224 | 24.885 | 42.058 | 1,024 | - |
|  | N: K/V-only >=16K | 771.941 | 51.397 | 24.885 | 41.263 | 1,024 | **+1.9268%** |
| Q/K norm + RoPE | C: S1 | 774.028 | 51.919 | 24.885 | 40.990 | 1,024 | - |
|  | N: S1 + Q/K | 776.849 | 51.492 | 24.885 | 41.073 | 1,024 | **-0.2034%** |
| Boundary norm | C: S1 | 781.277 | 51.681 | 24.885 | 40.885 | 1,024 | - |
|  | N: S1 + boundary | 782.377 | 51.616 | 24.944 | 40.891 | 1,024 | **-0.0130%** |
| Dual norm | C: S1 | 782.799 | 52.099 | 24.885 | 40.682 | 1,024 | - |
|  | N: S1 + dual | 786.854 | 51.911 | 24.885 | 40.661 | 1,024 | **+0.0505%** |
| Final QMV | C: S1 + dual | 716.905 | 49.979 | 24.885 | 43.465 | 1,024 | - |
|  | N: S1 + dual + QMV | 724.324 | 49.630 | 24.885 | 43.355 | 1,024 | **+0.2528%** |
| Source proposal | C: S1 + dual + QMV | 741.592 | 50.267 | 25.707 | 42.585 | 1,024 | - |
|  | N: final S4 stack | 757.719 | 52.035 | 25.707 | 41.400 | 1,024 | **+2.8631%** |

K/V has direct causal evidence beyond the aggregate wall delta. Its two timed
arms reduced prompt MTP-history append from 0.448-0.485 seconds to
0.112-0.116 seconds. Dual norm's corrected rerun engaged 877 calls in both
candidate arms and narrowly cleared the revised floor. C8 engaged 918 calls
per timed arm for each of E87 probe selection, fused row top-32, selected Q4
rerank, the Q precision island, and proposal selection; K and V islands each
engaged 1,207 times. Its timed control and candidate arms had the same matched
25.707 GiB peak because both route objects were resident for ABBA switching.

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
| Artifact/provenance | declared compact/island artifact | Retained by exact HF revision, raw LFS SHA-256, byte count, metadata, and tensor-shape checks. Runtime resolution is cache/path-only and never downloads implicitly. |

## Authoritative receipts

- Retained S1:
  `conditioned-s1-kv-ge16384-request-context-python-long16384in-1024out-t1-abba-2026-08-23.json`.
- Retained S2:
  `conditioned-r61-dual-stack-corrected-s1-python-long16384in-1024out-t1-abba-2026-08-23.json`.
- Retained S3:
  `c7-qmv-g32-g64-on-c3-c6-python16384in-1024out-t1-abba-2026-08-23.json`.
- Retained S4:
  `c8-source-proposal-on-c3-c6-c7-python16384in-1024out-t1-abba-2026-08-23.json`.
- Production S4 control-release verification:
  `final-s4-production-route-verify-2026-08-23.md` (25.041 GiB peak with the
  BF16 control body and full Q4 control head released before generation).
- Rejections: the corrected `c1-packed-qkv`, `c2-gdn-projection-pairs`,
  `chrono-r21`, and `chrono-r45` 16K receipts in the same directory.
- QMV numerical tie audit:
  `qmv-final-g32-g64-real-model-numeric-parity-2026-08-23.json`. This records
  the earlier uncorrected QMV attempt; it was rejected for performance, not
  parity. Corrected, group-aware C7 is retained by the receipt above.
- Static benchmark contract:
  `receipts/qwen38-challenge-port/control-contract-2026-08-23.md`.

The final route names only the four retained families. Rejected packed-QKV,
GDN-pair, Q/K-RoPE, and boundary-norm experiments are absent from the
production path.
