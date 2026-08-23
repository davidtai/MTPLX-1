# Qwen 3.8 Challenge Port Ledger

This ledger tracks the MTPLX disposition of the 82 accepted/promoted Qwen 3.8
MTP challenge submissions. The immutable source rows live in
`receipts/qwen38-challenge-port/yukon-accepted-2026-08-23.json`; the complete
row-by-row mechanism and skip decisions live in
`../../specs/2026-08-23-qwen38-challenge-port-design.md`.

## Source gate

- Yukon snapshot: 82 promoted submissions rendered 2026-08-23.
- Selection rule: `(score / previous promoted score - 1) * 100 > 0.10`.
- Qualifying source rows: 54.
- Challenge pin: `eb5eadc7a165047d4321ce883b9ff30894d8bd19`.
- MTPLX control pin: `bd4421567f9e16ce957c6ef97708b072dcd73937`.
- Reproduce: `python scripts/qwen38_challenge_inventory.py --check`.

The 54-row source gate is not the final port count. Resamples, no-ops,
challenge-only orchestration, already-present behavior, and mechanisms removed
or superseded later in the accepted sequence remain excluded. Below-threshold
rows are dependencies only when a surviving final mechanism requires them.

## Consolidated candidate ledger

| Candidate | Historical rows | Initial disposition | MTPLX gate | Receipt | Final status |
| --- | --- | --- | --- | --- | --- |
| Exact target top-2/readout reuse | 5, 6 | PORT | pending | pending | pending |
| Compact Q2 coarse/Q4 rerank proposal head | 10, 42, 46, 47, 67, 69, 71, 79, 82 | PORT plus row 46 dependency | pending | pending | pending |
| Final cross-row affine-4/group-64 QMV | 19, 34, 36, 39, 40, 41, 70, 78, 80 | PORT/dependency as one final family | pending | pending | pending |
| Projection/GDN/attention/norm fusions | 3, 13, 16, 18, 21, 30, 45, 60, 61 | PORT as final retained forms | pending | pending | pending |
| K/V-only history append and early submission | 11, 20 | PORT only for absent portions | pending | pending | pending |
| Compact-head precision-island artifact | 33, 36 | PORT artifact candidate | pending | pending | pending |
| Final depth/warm calibration | 11, 38, 59 | PORT into existing controller only | pending | pending | pending |

## Skip summary

- **ALREADY:** committed history/prompt streaming, exact replay, generalized
  verify, cost-model infrastructure, Q4/group-64 draft head, and current
  NAX/compiled-verify behavior remain MTPLX-owned and unchanged.
- **WEAK:** 27 rows at or below the threshold, controlled resamples, and
  no-op/noise rows have no performance claim.
- **SUPERSEDED:** nine historical mechanisms are represented only by their
  surviving descendants or are omitted after an accepted reversal.
- **CHALLENGE-ONLY:** Swift worker plumbing, declared-head staging, and Metal
  command-buffer/residency policy do not map to the Python MTPLX runtime.
- **DEPENDENCY:** seven below-threshold or intermediate rows may inform a final
  candidate but receive neither an independent port nor an independent win.

Production code is retained only after exactness, one-cycle, 64-token, and
512-token matched gates against the unchanged control. A regression rejects the
candidate at the first failing gate.

## Control status

The static control contract is captured in
`receipts/qwen38-challenge-port/control-contract-2026-08-23.md`. No real-model
baseline was taken during inventory freeze because `/tmp/mtplx-gpu-exclusive.lock`
was owned by an unrelated benchmark. That is a deferred measurement, not a
zero or inferred baseline.
