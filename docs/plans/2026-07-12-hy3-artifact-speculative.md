# Hy3 Artifact and Speculative Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide independently whether GPU-oriented expert packing, KV/cache budget exchange, hint-only route prefetch, or a lower-bit cold tier justifies implementation.
**Architecture:** Build on PR 2, but apply the promotion premise before writing selectors or runtime code. Existing isolated results count when their mechanism and claim boundary match #31. No sub-5% arms may be combined.
**Tech Stack:** Checked-in route/kernel/runtime benchmark evidence, manifest contracts, MLX capability inspection, and pytest verification.
**Premise update:** #29 and #30 provide the required baseline. A clean-commit capacity-102 probe now rejects the 144-byte same-byte packed layout: +0.023% weighted speedup, 95% CI [-0.444%, +0.491%]. Existing evidence also rejects the exact measured 75-to-100-slot reinvestment and the current MTP speed configuration. The trace-local route-recall data is insufficient to justify runtime promotion, not evidence that every hint-only predictor fails. Track 4 lacks a valid quality/artifact contract. The detailed evidence map is `benchmarks/results/hy3-artifact-speculative-issue31-20260713.{md,json}`.

---

### Task 1: Gate a synthetic v2 codec with a packed-stream probe

- [x] Recover the measured same-byte stream floor and single-dispatch roofline from commit `2671ba7`.
- [x] Correct the claim boundary: `0.1977 / 0.1871 - 1 = 5.665%` isolated headroom, while the direct `0.0106 ms/layer * 79` delta is only about 0.247% of the #30 end-to-end baseline.
- [x] Record that the only unmeasured 144-byte packed layout reduces streams from nine to three but removes no bytes.
- [x] Add behavior-locking tests for the exact four-group packed byte layout and benchmark result contract.
- [x] Run one standalone, raw-byte/checksum-equivalent, paired split-versus-packed Metal stream-floor probe at the production gate/up and down shapes.
- [x] Stop before an exact packed-QMV probe: the weighted upper 95% bound is +0.491%, so the floor cannot reach the fixed +5% gate. Do not build a codec or 161 GB sidecar.

### Task 2: Decide the KV-to-expert budget exchange

- [x] Audit the repaired PR #12 Q8 accounting, attestation, memory, and performance evidence.
- [x] Reject the 75-to-100-slot reinvestment arm: decode -78.97%, p95 +351.5%, peak MLX +20.97 GB.
- [x] Preserve the claim boundary: both measured arms used physical Q8, so this is not Q8-versus-BF16 quality evidence.
- [x] Keep fixed-slot BF16-versus-Q8 representation work open until it has a true long-context attention/quality method.
- [x] Do not evaluate Q4 KV before that Q8 method exists.

### Task 3: Gate an authoritative-router-safe prefetch API offline

- [x] Audit prior-token and prompt-trained top-8/16/32 route analysis with byte amplification.
- [x] Audit the isolated Hy3 Q4 MTP acceptance and AR/MTP speed gate.
- [x] Record the boundary: same-trace top-16 recalls 52.72% at 2x nominal requests and top-32 recalls 67.91% at 4x, while only the current MTP configuration has a directly measured speed no-go (-37.05% decode).
- [x] Keep the trunk router authoritative and stop before adding a runtime prefetch API.
- [x] Leave the general hint-only track open and require a future predictor to clear an offline held-out, physical-byte-amplification-aware, net-latency premise before runtime TDD begins.

### Task 4: Define the lower-bit prerequisite boundary

- [x] Confirm local MLX affine kernel capability separately from the deployed Hy3 Q4-only runtime contract.
- [x] Calculate, but do not present as measured, Q3 (-22.22%) and Q2 (-44.44%) record-size reductions.
- [x] Record the missing artifact identity and perplexity/reasoning/tool-use/long-generation quality prerequisites.
- [x] Defer implementation; no lower-bit artifact may claim the exact Q4 model identity or silently dequantize.
- [x] Specify a one-layer Q3 held-out pilot as the smallest valid reopening gate.

### Task 5: Verify and publish the stacked #31 experiment PR

- [x] Reuse only independently labeled experiments whose mechanism matches the #31 arm.
- [x] Run full pytest and changed-file Ruff.
- [x] Save a machine-readable decision payload and a human-readable per-track evidence map.
- [x] Document go/no-go independently; do not combine sub-5% arms into one claimed win or close the issue while open tracks remain.
- [x] Push `experiment/hy3-artifact-speculative` and open draft PR #37 against `experiment/hy3-record-native-exec`, linking #31 without closing it.
